from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import Mol
import pubchempy as pcp
import selfies as sf
from google.generativeai import GenerativeModel
import google.generativeai as genai
import time
import logging
import re
import json
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import hashlib
import pickle
import os
from time import sleep
import random
from datetime import datetime

# Suppress RDKit warnings
RDLogger.DisableLog('rdApp.*')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Gemini API
GEMINI_API_KEY = "AIzaSyCf8gvsefTQSLxVAV5hcn2ysHxKB9LXyoc"
genai.configure(api_key=GEMINI_API_KEY)
model = GenerativeModel("gemini-2.5-pro-exp-03-25")

# Cache configuration
CACHE_DIR = "gemini_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_EXPIRY_DAYS = 7

class APIManager:
    def __init__(self, calls_per_minute=10, daily_limit=500):
        self.calls_per_minute = calls_per_minute
        self.daily_limit = daily_limit
        self.last_call_time = 0
        self.call_count = 0
        self.total_calls = 0
        self.daily_usage_file = os.path.join(CACHE_DIR, "api_usage.json")
        self._load_daily_usage()
        
    def _load_daily_usage(self):
        """Load daily usage from file"""
        if os.path.exists(self.daily_usage_file):
            try:
                with open(self.daily_usage_file, 'r') as f:
                    data = json.load(f)
                    if data.get('date') == datetime.now().strftime("%Y-%m-%d"):
                        self.total_calls = data.get('count', 0)
            except Exception:
                pass

    def _save_daily_usage(self):
        """Save current daily usage"""
        try:
            with open(self.daily_usage_file, 'w') as f:
                json.dump({
                    'date': datetime.now().strftime("%Y-%m-%d"),
                    'count': self.total_calls
                }, f)
        except Exception:
            pass

    def _get_cache_key(self, prompt: str) -> str:
        return hashlib.md5(prompt.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> str:
        return os.path.join(CACHE_DIR, f"{key}.pkl")
    
    def get_cached_response(self, prompt: str):
        """Get cached response if available and fresh"""
        key = self._get_cache_key(prompt)
        cache_file = self._get_cache_path(key)
        
        if os.path.exists(cache_file):
            file_age = time.time() - os.path.getmtime(cache_file)
            if file_age < CACHE_EXPIRY_DAYS * 86400:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        return None
    
    def cache_response(self, prompt: str, response):
        """Cache response with error handling"""
        try:
            key = self._get_cache_key(prompt)
            cache_file = self._get_cache_path(key)
            with open(cache_file, 'wb') as f:
                pickle.dump(response, f)
        except Exception as e:
            logger.warning(f"Caching failed: {e}")

    def make_api_call(self, prompt: str, is_critical=False):
        """Make API call with enhanced rate limiting"""
        # Check daily limit first
        if self.total_calls >= self.daily_limit and not is_critical:
            logger.warning("Daily API limit reached - using fallback methods")
            return None
            
        # Rate limiting with adaptive backoff
        elapsed = time.time() - self.last_call_time
        min_interval = 60 / self.calls_per_minute
        
        if elapsed < min_interval:
            sleep_time = min_interval - elapsed + random.uniform(0.1, 0.5)
            time.sleep(sleep_time)
        
        try:
            response = model.generate_content(prompt)
            self.last_call_time = time.time()
            self.call_count += 1
            self.total_calls += 1
            self._save_daily_usage()
            
            # Dynamic rate adjustment
            if self.call_count % 10 == 0:
                remaining = max(0, self.daily_limit - self.total_calls)
                logger.info(f"API calls: {self.total_calls}/{self.daily_limit} today. {remaining} remaining")
                
                # Auto-throttle if approaching limit
                if remaining < 100:
                    self.calls_per_minute = max(2, self.calls_per_minute // 2)
                    logger.warning(f"Reducing rate to {self.calls_per_minute} calls/minute")
            
            if response.text:
                self.cache_response(prompt, response.text)
                return response.text
        except Exception as e:
            logger.error(f"API Error: {e}")
            # Exponential backoff on errors
            time.sleep(min(5, 2 ** (self.call_count // 10)))
        return None
    
    def reset_counter(self):
        """Reset minute counter but preserve daily total"""
        self.call_count = 0

class DrugGenerator:
    def __init__(self, latent_dim=50, max_length=300):
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.char_to_int, self.int_to_char = self._create_tokenizer()
        self.start_time = time.time()
        self.api_manager = APIManager(calls_per_minute=10, daily_limit=500)
        self.training_compounds = self._get_robust_training_set()
        
    def _get_robust_training_set(self) -> List[str]:
        """Return a comprehensive set of drug molecules for training"""
        return [
            # Pain/Inflammation
            'Aspirin', 'Ibuprofen', 'Naproxen', 'Celecoxib', 'Diclofenac',
            # CNS
            'Diazepam', 'Sertraline', 'Fluoxetine', 'Venlafaxine', 'Duloxetine',
            # Cardiovascular
            'Atorvastatin', 'Simvastatin', 'Lisinopril', 'Metoprolol', 'Amlodipine',
            # Antibiotics
            'Amoxicillin', 'Ciprofloxacin', 'Azithromycin', 'Doxycycline',
            # Metabolic
            'Metformin', 'Glipizide', 'Pioglitazone', 'Empagliflozin',
            # Others
            'Omeprazole', 'Loratadine', 'Diphenhydramine', 'Tramadol'
        ]
    
    def _create_tokenizer(self) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Create tokenizer with extended SELFIES alphabet"""
        valid_chars = list(sf.get_semantic_robust_alphabet())
        valid_chars.extend(['[Br]', '[Cl]', '[=O]', '[N+]', '[O-]'])
        return {c: i for i, c in enumerate(valid_chars)}, {i: c for i, c in enumerate(valid_chars)}

    def validate_medical_prompt(self, prompt: str) -> bool:
        """Check if prompt is medically relevant"""
        medical_keywords = [
            'drug', 'medicine', 'pharmaceutical', 'treatment', 'therapy',
            'disease', 'illness', 'condition', 'symptom', 'pain',
            'infection', 'virus', 'bacteria', 'cancer', 'diabetes',
            'blood pressure', 'cholesterol', 'antibiotic', 'antiviral',
            'anti-inflammatory', 'analgesic', 'antidepressant', 'inhibitor',
            'agonist', 'antagonist', 'receptor', 'enzyme', 'blocker'
        ]
        prompt = prompt.lower()
        return any(keyword in prompt for keyword in medical_keywords)

    def _quick_validate(self, tokens: str) -> Optional[str]:
        """More robust validation with multiple attempts"""
        for _ in range(3):
            try:
                smiles = sf.decoder(tokens)
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    mol = AllChem.RemoveHs(mol)
                    Chem.SanitizeMol(mol)
                    return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
            except Exception:
                try:
                    mol = Chem.MolFromSmiles(tokens)
                    if mol:
                        return Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
                except Exception:
                    continue
        return None

    def _encode_data(self, smiles_list: List[str]) -> np.ndarray:
        """Convert SELFIES strings to one-hot encoded matrix"""
        X = np.zeros((len(smiles_list), self.max_length, len(self.char_to_int)), dtype=np.float32)
        valid_indices = []
        
        for i, selfies in enumerate(tqdm(smiles_list, desc="Encoding Molecules")):
            try:
                symbols = list(sf.split_selfies(selfies))[:self.max_length]
                for t, char in enumerate(symbols):
                    if char in self.char_to_int:
                        X[i, t, self.char_to_int[char]] = 1.0
                valid_indices.append(i)
            except Exception:
                continue
        
        return X[valid_indices]

    def train(self, epochs: int = 100) -> None:
        """Train VAE with enhanced parameters"""
        print("Loading training data...")
        selfies_list = []
        
        for name in tqdm(self.training_compounds, desc="Loading Compounds"):
            try:
                compounds = pcp.get_compounds(name, 'name')
                if compounds:
                    smiles = compounds[0].isomeric_smiles
                    mol = Chem.MolFromSmiles(smiles)
                    if mol and '.' not in smiles:
                        selfies = sf.encoder(smiles)
                        if Chem.MolFromSmiles(sf.decoder(selfies)):
                            selfies_list.append(selfies)
            except Exception as e:
                logger.warning(f"Failed to process {name}: {e}")
                continue
        
        if len(selfies_list) < 10:
            raise ValueError("Insufficient valid training data")
        
        print(f"\nPreparing training data from {len(selfies_list)} compounds...")
        X = self._encode_data(selfies_list)
        
        # Enhanced Encoder
        inputs = keras.Input(shape=X.shape[1:])
        x = layers.Flatten()(inputs)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(256, activation='relu')(x)
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)
        
        z = layers.Lambda(
            lambda p: p[0] + tf.exp(0.5 * p[1]) * tf.random.normal(tf.shape(p[0])),
            name='z'
        )([z_mean, z_log_var])
        
        # Enhanced Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(256, activation='relu')(latent_inputs)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dense(np.prod(X.shape[1:]), activation='softmax')(x)
        outputs = layers.Reshape(X.shape[1:])(x)
        
        # Combine and compile
        self.encoder = keras.Model(inputs, [z_mean, z_log_var, z])
        self.decoder = keras.Model(latent_inputs, outputs)
        self.vae = keras.Model(inputs, self.decoder(self.encoder(inputs)[2]))
        
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-3,
            decay_steps=1000,
            decay_rate=0.9)
        
        self.vae.compile(optimizer=keras.optimizers.Adam(lr_schedule), 
                        loss='binary_crossentropy')
        
        print("\nStarting training...")
        self.vae.fit(
            X, X,
            epochs=epochs,
            batch_size=64,
            shuffle=True,
            verbose=0,
            callbacks=[
                EpochProgressCallback(epochs),
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
            ]
        )

    def generate_molecules(self, num_molecules: int = 30) -> List[str]:
        """Generate molecules with multiple attempts"""
        print("\nGenerating molecular structures:")
        molecules = []
        start_time = time.time()
        attempts = 0
        max_attempts = num_molecules * 5
        
        with tqdm(total=num_molecules, desc="Valid Molecules Generated") as pbar:
            while len(molecules) < num_molecules and attempts < max_attempts:
                try:
                    if attempts % 10 == 0:
                        latent = np.random.uniform(-2, 2, size=(1, self.latent_dim))
                    else:
                        latent = np.random.normal(scale=0.5, size=(1, self.latent_dim))
                    
                    decoded = self.decoder.predict(latent, verbose=0)
                    tokens = "".join([self.int_to_char[i] 
                                    for i in np.argmax(decoded[0], axis=-1) 
                                    if i in self.int_to_char])
                    
                    smiles = self._quick_validate(tokens)
                    if smiles and smiles not in molecules:
                        molecules.append(smiles)
                        pbar.update(1)
                except Exception:
                    pass
                finally:
                    attempts += 1
                
                if time.time() - start_time > 60:
                    break
        
        print(f"Generated {len(molecules)} valid molecules from {attempts} attempts")
        return molecules

    def _extract_smiles(self, text: str) -> Optional[str]:
        """More robust SMILES extraction from text"""
        text = re.sub(r'^```[a-z]*\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n```$', '', text, flags=re.MULTILINE)
        text = text.strip('`').strip()
        
        mol = Chem.MolFromSmiles(text)
        if mol:
            return Chem.MolToSmiles(mol, canonical=True)
            
        pattern = r'(?:^|\s)((?:C|O|N|S|Cl|Br|F|I)(?:[a-z0-9=#@$%\[\]()+\-/\\]+))(?:$|\s)'
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                mol = Chem.MolFromSmiles(match)
                if mol:
                    return Chem.MolToSmiles(mol, canonical=True)
            except Exception:
                continue
        return None

    def optimize_with_gemini(self, smiles: str, max_retries: int = 1) -> str:
        """Optimize molecule with multiple strategies"""
        original_smiles = smiles
        
        cache_key = f"optimize_{smiles}"
        cached = self.api_manager.get_cached_response(cache_key)
        if cached and Chem.MolFromSmiles(cached):
            return cached
            
        if self.api_manager.total_calls < self.api_manager.daily_limit:
            prompt = f"""Optimize this molecule for drug-likeness while maintaining therapeutic potential.
                      Consider improving bioavailability, metabolic stability, and target binding.
                      Return ONLY the SMILES string without any additional text or formatting: {smiles}"""
            
            for attempt in range(max_retries + 1):
                try:
                    response_text = self.api_manager.make_api_call(prompt)
                    if response_text:
                        optimized = self._extract_smiles(response_text)
                        if optimized and Chem.MolFromSmiles(optimized):
                            self.api_manager.cache_response(cache_key, optimized)
                            return optimized
                except Exception as e:
                    logger.warning(f"Optimization attempt {attempt + 1} failed: {e}")
                    time.sleep(1)
        
        return self._rule_based_optimization(original_smiles)

    def _rule_based_optimization(self, smiles: str) -> str:
        """Chemical-aware optimization when API is unavailable"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if not mol:
                return smiles
                
            modified = False
            
            if Descriptors.MolWt(mol) > 500:
                for atom in mol.GetAtoms():
                    if atom.GetDegree() == 1 and atom.GetAtomicNum() == 6:
                        mol.RemoveAtom(atom.GetIdx())
                        modified = True
                        break
            
            logp = Descriptors.MolLogP(mol)
            if logp > 5:
                for atom in mol.GetAtoms():
                    if atom.GetIsAromatic() and atom.GetAtomicNum() == 6:
                        atom.SetAtomicNum(7)
                        modified = True
                        break
            
            rings = rdMolDescriptors.CalcNumRings(mol)
            if rings == 0 and mol.GetNumAtoms() > 5:
                ed = Chem.EditableMol(mol)
                ed.AddBond(0, min(4, mol.GetNumAtoms()-1), Chem.BondType.SINGLE)
                modified = True
            elif rings > 4:
                for bond in mol.GetBonds():
                    if bond.IsInRing():
                        mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                        modified = True
                        break
            
            if modified:
                new_smiles = Chem.MolToSmiles(mol)
                if new_smiles and Chem.MolFromSmiles(new_smiles):
                    return new_smiles
        except Exception:
            pass
        return smiles

    def optimize_batch(self, molecules: List[str]) -> List[str]:
        """Optimize molecules with smart batching"""
        print("\nOptimizing molecules:")
        
        optimized = []
        remaining = []
        
        for smi in molecules:
            cache_key = f"optimize_{smi}"
            cached = self.api_manager.get_cached_response(cache_key)
            if cached and Chem.MolFromSmiles(cached):
                optimized.append(cached)
            else:
                remaining.append(smi)
        
        if remaining and self.api_manager.total_calls < self.api_manager.daily_limit:
            batch_size = min(10, self.api_manager.daily_limit - self.api_manager.total_calls)
            for smi in tqdm(remaining[:batch_size], desc="API Optimization"):
                opt = self.optimize_with_gemini(smi)
                optimized.append(opt)
        
        if len(optimized) < len(molecules):
            remaining = molecules[len(optimized):]
            for smi in tqdm(remaining, desc="Rule-Based Optimization"):
                optimized.append(self._rule_based_optimization(smi))
        
        return optimized[:len(molecules)]

    def get_constraints(self, description: str) -> Dict:
        """Get constraints with medical validation"""
        if not self.validate_medical_prompt(description):
            raise ValueError("Input must be medically relevant")
        
        cache_key = f"constraints_{description}"
        cached = self.api_manager.get_cached_response(cache_key)
        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                pass
            
        prompt = f"""Convert this drug description into molecular properties. 
                  Return ONLY a JSON dictionary with these exact keys:
                  - molecular_weight_range: [min, max]
                  - logp_range: [min, max]
                  - hba_range: [min, max]
                  - hbd_range: [min, max]
                  - rotatable_bonds: max
                  - rings_range: [min, max]
                  - psa_range: [min, max]
                  - aromatic_rings: [min, max]
                  
                  Description: {description}"""
        
        response_text = self.api_manager.get_cached_response(prompt)
        if not response_text:
            response_text = self.api_manager.make_api_call(prompt)
            if not response_text:
                return self._get_default_constraints()
            
        try:
            json_str = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_str:
                constraints = json.loads(json_str.group())
                if self._validate_constraints(constraints):
                    self.api_manager.cache_response(cache_key, json_str.group())
                    return constraints
        except json.JSONDecodeError:
            pass
            
        return self._get_default_constraints()

    def _validate_constraints(self, constraints: Dict) -> bool:
        """Validate constraints structure"""
        required_keys = {
            'molecular_weight_range': {'type': list, 'length': 2},
            'logp_range': {'type': list, 'length': 2},
            'hba_range': {'type': list, 'length': 2},
            'hbd_range': {'type': list, 'length': 2},
            'rotatable_bonds': {'type': int},
            'rings_range': {'type': list, 'length': 2},
            'psa_range': {'type': list, 'length': 2},
            'aromatic_rings': {'type': list, 'length': 2}
        }
        
        for key, requirements in required_keys.items():
            if key not in constraints:
                return False
            if not isinstance(constraints[key], requirements['type']):
                return False
            if requirements['type'] == list and 'length' in requirements:
                if len(constraints[key]) != requirements['length']:
                    return False
        return True

    def _get_default_constraints(self) -> Dict:
        """Relaxed default constraints"""
        return {
            'molecular_weight_range': [150, 600],  # Wider range
            'logp_range': [-1, 6],  # More flexible
            'hba_range': [1, 12],  # Expanded
            'hbd_range': [0, 7],  # Allow zero HBD
            'rotatable_bonds': 10,  # Increased
            'rings_range': [0, 5],  # Allow no rings
            'psa_range': [10, 150],  # Broader range
            'aromatic_rings': [0, 4]  # Allow no aromatic rings
        }

def passes_filters(smiles: str, constraints: Dict) -> bool:
    """Lenient filtering that checks basic validity"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return False
        
        # Basic validity checks only
        return all([
            mol.GetNumAtoms() > 3,  # Minimum size
            mol.GetNumBonds() > 1,  # Must have connections
            Descriptors.MolWt(mol) < 1000,  # Absolute maximum
            Descriptors.MolLogP(mol) < 10,  # Absolute maximum
            Descriptors.NumHAcceptors(mol) < 20,  # Absolute maximum
            Descriptors.NumHDonors(mol) < 10  # Absolute maximum
        ])
    except Exception:
        return False

class EpochProgressCallback(keras.callbacks.Callback):
    def __init__(self, total_epochs):
        self.epoch_bar = tqdm(total=total_epochs, desc="Training Epochs", unit="epoch")
        
    def on_epoch_end(self, epoch, logs=None):
        self.epoch_bar.update(1)
        self.epoch_bar.set_postfix({
            'loss': f"{logs['loss']:.4f}",
            'val_loss': f"{logs.get('val_loss', 'NA')}"
        })
        
    def on_train_end(self, logs=None):
        self.epoch_bar.close()

def count_aromatic_rings(mol):
    """Count the number of aromatic rings in a molecule"""
    try:
        # Get all rings in the molecule
        ring_info = mol.GetRingInfo()
        atom_rings = ring_info.AtomRings()
        
        # Count aromatic rings
        aromatic_rings = 0
        for ring in atom_rings:
            is_aromatic = True
            for atom_idx in ring:
                if not mol.GetAtomWithIdx(atom_idx).GetIsAromatic():
                    is_aromatic = False
                    break
            if is_aromatic:
                aromatic_rings += 1
        return aromatic_rings
    except Exception:
        return 0

# Initialize the drug generator
drug_generator = DrugGenerator()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'api_ready': True})

@app.route('/api/generate-drugs', methods=['POST'])
def generate_drugs():
    try:
        data = request.get_json()
        description = data.get('description', '').strip()
        
        if not description:
            return jsonify({'error': 'Please provide a drug description'}), 400
        
        if not drug_generator.validate_medical_prompt(description):
            return jsonify({'error': 'Input must be medically relevant'}), 400
        
        # Get constraints
        constraints = drug_generator.get_constraints(description)
        
        # Train the model (only if not already trained)
        try:
            drug_generator.train(epochs=100)
        except Exception as e:
            return jsonify({'error': f'Training failed: {str(e)}'}), 500
        
        # Generate molecules
        raw_molecules = drug_generator.generate_molecules(30)
        optimized_molecules = drug_generator.optimize_batch(raw_molecules)
        
        # Filter and prepare results
        final_molecules = []
        for smi in optimized_molecules:
            if passes_filters(smi, constraints):
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    final_molecules.append({
                        'smiles': smi,
                        'molecular_weight': Descriptors.MolWt(mol),
                        'logp': Descriptors.MolLogP(mol),
                        'hba': Descriptors.NumHAcceptors(mol),
                        'hbd': Descriptors.NumHDonors(mol),
                        'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
                        'rings': rdMolDescriptors.CalcNumRings(mol),
                        'psa': Descriptors.TPSA(mol),
                        'aromatic_rings': count_aromatic_rings(mol)
                    })
        
        return jsonify({
            'molecules': final_molecules[:10],  # Return top 10 molecules
            'constraints': constraints
        })
        
    except Exception as e:
        logger.error(f"Error in generate_drugs: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)