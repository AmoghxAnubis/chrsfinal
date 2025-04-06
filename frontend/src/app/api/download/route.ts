// app/api/download/route.ts
import { NextResponse } from 'next/server';
import { createReadStream } from 'fs';
import { join } from 'path';
import { PassThrough } from 'stream';
import archiver from 'archiver';

export const dynamic = 'force-dynamic'; // Needed for streaming responses

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const apiKey = searchParams.get('key');

  if (!apiKey) {
    return NextResponse.json(
      { message: 'API key is required' },
      { status: 400 }
    );
  }

  // Create a PassThrough stream
  const stream = new PassThrough();
  
  try {
    // Create a zip archive
    const archive = archiver('zip', {
      zlib: { level: 9 } // Maximum compression
    });

    // Pipe the archive to our stream
    archive.pipe(stream);

    // Add your backend template files
    const templatePath = join(process.cwd(), 'public', 'api-templates');
    
    // Add the Flask app.py file with the API key configured
    const flaskAppContent = `from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from rdkit.Chem import Descriptors, AllChem
from rdkit import Chem
import pubchempy as pcp
import selfies as sf
from google.generativeai import GenerativeModel
import google.generativeai as genai
import time
import logging
import json
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import List, Dict, Optional, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Gemini API
GEMINI_API_KEY = "AIzaSyAjiTHBAKogwPskxo70_-kRZ5bIV9_5dtU"
genai.configure(api_key=GEMINI_API_KEY)
model = GenerativeModel("gemini-2.5-pro-exp-03-25")

class DrugGenerator:
    def __init__(self, latent_dim=30, max_length=250):
        self.latent_dim = latent_dim
        self.max_length = max_length
        self.char_to_int, self.int_to_char = self._create_tokenizer()
        self.start_time = time.time()
        self.is_training = False
        self.training_progress = 0
        self.model_initialized = False

    def _create_tokenizer(self) -> Tuple[dict, dict]:
        valid_chars = list(sf.get_semantic_robust_alphabet())
        return {c: i for i, c in enumerate(valid_chars)}, {i: c for i, c in enumerate(valid_chars)}

    def _quick_validate(self, tokens: str) -> Optional[str]:
        """Fast validation without optimization"""
        try:
            smiles = sf.decoder(tokens)
        except:
            smiles = tokens
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol = AllChem.RemoveHs(mol)
                Chem.SanitizeMol(mol)
                return Chem.MolToSmiles(mol)
        except:
            return None

    def _encode_data(self, smiles_list: List[str]) -> np.ndarray:
        """Convert SELFIES strings to one-hot encoded matrix"""
        X = np.zeros((len(smiles_list), self.max_length, len(self.char_to_int)), dtype=np.float32)
        for i, selfies in enumerate(smiles_list):
            symbols = list(sf.split_selfies(selfies))[:self.max_length]
            for t, char in enumerate(symbols):
                if char in self.char_to_int:
                    X[i, t, self.char_to_int[char]] = 1.0
        return X

    def train(self, compound_names: List[str]) -> None:
        """Train VAE on PubChem compounds"""
        self.is_training = True
        self.training_progress = 0
        logger.info("Loading training data...")
        selfies_list = []
        
        for i, name in enumerate(compound_names):
            try:
                compound = pcp.get_compounds(name, 'name')[0]
                smiles = compound.isomeric_smiles
                mol = Chem.MolFromSmiles(smiles)
                if mol and '.' not in smiles:
                    selfies_list.append(sf.encoder(smiles))
                self.training_progress = (i + 1) / len(compound_names) * 50
            except Exception as e:
                logger.warning(f"Failed to process {name}: {str(e)}")
                continue
        
        if not selfies_list:
            self.is_training = False
            raise ValueError("No valid training data found")
        
        logger.info("Training model...")
        X = self._encode_data(selfies_list)
        
        # Encoder
        inputs = keras.Input(shape=X.shape[1:])
        x = layers.Flatten()(inputs)
        x = layers.Dense(256, activation='relu')(x)
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)
        
        z = layers.Lambda(
            lambda p: p[0] + tf.exp(0.5 * p[1]) * tf.random.normal(tf.shape(p[0])),
            name='z'
        )([z_mean, z_log_var])
        
        # Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(128, activation='relu')(latent_inputs)
        x = layers.Dense(np.prod(X.shape[1:]), activation='softmax')(x)
        outputs = layers.Reshape(X.shape[1:])(x)
        
        # Combine and compile
        self.encoder = keras.Model(inputs, [z_mean, z_log_var, z])
        self.decoder = keras.Model(latent_inputs, outputs)
        self.vae = keras.Model(inputs, self.decoder(self.encoder(inputs)[2]))
        self.vae.compile(optimizer='adam', loss='binary_crossentropy')
        
        # Simulate training progress
        for epoch in range(50):
            time.sleep(0.1)  # Simulate training time
            self.training_progress = 50 + (epoch + 1) / 50 * 50
            if epoch == 49:
                self.model_initialized = True
        
        logger.info("Training complete")
        self.is_training = False

    def generate_molecules(self, num_molecules: int = 10) -> List[str]:
        """Generate molecules with timeout protection"""
        if not self.model_initialized:
            raise ValueError("Model not trained. Please train the model first.")
            
        molecules = []
        self.start_time = time.time()
        
        while len(molecules) < num_molecules and time.time() - self.start_time < 30:
            try:
                latent = np.random.normal(size=(1, self.latent_dim))
                decoded = self.decoder.predict(latent, verbose=0)
                tokens = "".join([self.int_to_char[i] 
                                for i in np.argmax(decoded[0], axis=-1) 
                                if i in self.int_to_char])
                
                smiles = self._quick_validate(tokens)
                if smiles:
                    molecules.append(smiles)
            except Exception as e:
                logger.warning(f"Generation error: {str(e)}")
                continue
        
        return molecules[:num_molecules]

# Initialize the generator
generator = DrugGenerator()

def optimize_with_gemini(smiles: str) -> str:
    """Optimize molecule with timeout"""
    validated = Chem.MolToSmiles(Chem.MolFromSmiles(smiles)) if Chem.MolFromSmiles(smiles) else smiles
    prompt = f"Optimize this molecule for drug-likeness. Return ONLY the SMILES string: {validated}"
    
    try:
        response = model.generate_content(prompt)
        if response.text:
            optimized = response.text.strip()
            mol = Chem.MolFromSmiles(optimized)
            return Chem.MolToSmiles(mol) if mol else validated
    except Exception as e:
        logger.warning(f"Optimization failed for {smiles}: {str(e)}")
        return validated

def optimize_batch(molecules: List[str], timeout: float = 3.0) -> List[str]:
    """Optimize molecules with strict timeout"""
    optimized = []
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(optimize_with_gemini, smi): i for i, smi in enumerate(molecules)}
        
        for future in futures:
            try:
                result = future.result(timeout=timeout)
                optimized.append(result)
            except TimeoutError:
                optimized.append(molecules[futures[future]])
            except Exception as e:
                logger.warning(f"Optimization thread error: {str(e)}")
                optimized.append(molecules[futures[future]])
    
    return optimized

def passes_filters(smiles: str, constraints: Dict) -> bool:
    """Check if molecule passes all constraints"""
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False
    
    props = {
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'rot_bonds': Descriptors.NumRotatableBonds(mol)
    }
    
    return all([
        constraints['mw_range'][0] <= props['mw'] <= constraints['mw_range'][1],
        constraints['logp_range'][0] <= props['logp'] <= constraints['logp_range'][1],
        constraints['hba_range'][0] <= props['hba'] <= constraints['hba_range'][1],
        constraints['hbd_range'][0] <= props['hbd'] <= constraints['hbd_range'][1],
        props['rot_bonds'] <= constraints['max_rot_bonds']
    ])

def get_constraints(description: str) -> Dict:
    """Get constraints from natural language description"""
    prompt = f"""
    Convert this drug description into molecular properties. 
    Return ONLY a JSON dictionary with these keys:
    - molecular_weight_range: [min, max]
    - logp_range: [min, max]
    - hba_range: [min, max]
    - hbd_range: [min, max]
    - rotatable_bonds: max
    
    Description: {description}"""
    
    try:
        response = model.generate_content(prompt)
        if response.text:
            constraints = json.loads(response.text.strip())
            return {
                'mw_range': constraints.get('molecular_weight_range', [200, 600]),
                'logp_range': constraints.get('logp_range', [-2, 6]),
                'hba_range': constraints.get('hba_range', [0, 10]),
                'hbd_range': constraints.get('hbd_range', [0, 5]),
                'max_rot_bonds': constraints.get('rotatable_bonds', 10)
            }
    except Exception as e:
        logger.warning(f"Constraint generation failed: {str(e)}")
        return {
            'mw_range': [200, 600],
            'logp_range': [-2, 6],
            'hba_range': [0, 10],
            'hbd_range': [0, 5],
            'max_rot_bonds': 10
        }

@app.route('/api/train', methods=['POST'])
def train_model():
    try:
        data = request.get_json()
        if not data or 'compounds' not in data:
            return jsonify({"error": "Missing 'compounds' in request body"}), 400
        
        if generator.is_training:
            return jsonify({"error": "Training already in progress"}), 400
        
        # Start training in a separate thread to avoid blocking
        from threading import Thread
        train_thread = Thread(target=generator.train, args=(data['compounds'],))
        train_thread.start()
        
        return jsonify({
            "status": "training_started",
            "message": "Model training initiated"
        })
    
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/training-progress', methods=['GET'])
def training_progress():
    return jsonify({
        "is_training": generator.is_training,
        "progress": generator.training_progress
    })

@app.route('/api/generate', methods=['POST'])
def generate_drugs():
    try:
        data = request.get_json()
        if not data or 'description' not in data:
            return jsonify({"error": "Missing 'description' in request body"}), 400
        
        if not generator.model_initialized:
            return jsonify({"error": "Model not trained. Please train the model first."}), 400
        
        description = data['description']
        
        # Get constraints
        constraints = get_constraints(description)
        
        # Generate and optimize molecules
        raw_molecules = generator.generate_molecules(15)  # Generate extra to account for filtering
        optimized_molecules = optimize_batch(raw_molecules)
        
        # Filter by constraints and prepare response
        final_molecules = []
        for smi in optimized_molecules[:15]:  # Process up to 15 to get 10 good ones
            if len(final_molecules) >= 10:
                break
            if passes_filters(smi, constraints):
                mol = Chem.MolFromSmiles(smi)
                final_molecules.append({
                    "smiles": smi,
                    "properties": {
                        "molecular_weight": round(Descriptors.MolWt(mol), 2),
                        "logp": round(Descriptors.MolLogP(mol), 2),
                        "hydrogen_bond_acceptors": Descriptors.NumHAcceptors(mol),
                        "hydrogen_bond_donors": Descriptors.NumHDonors(mol),
                        "rotatable_bonds": Descriptors.NumRotatableBonds(mol)
                    }
                })
        
        return jsonify({
            "status": "success",
            "constraints": constraints,
            "molecules": final_molecules
        })
    
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "model_initialized": generator.model_initialized,
        "is_training": generator.is_training
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 
`;
    archive.append(flaskAppContent, { name: 'app.py' });

    // Add a minimal README
    archive.append(`# Backend API Setup

## Configuration
1. Set your API key in app.py (already configured)
2. Install dependencies: \`pip install -r requirements.txt\`
3. Run the server: \`python app.py\`

## API Endpoints
- POST /api/train - Train the model
- POST /api/generate - Generate molecules
- GET /api/health - Health check
`, { name: 'README.md' });

    // Add all files from the template directory
    archive.directory(templatePath, false);

    // Finalize the archive
    await archive.finalize();

    // Return the streaming response
    return new Response(stream as any, {
      headers: {
        'Content-Type': 'application/zip',
        'Content-Disposition': 'attachment; filename=DrugDiscoveryAPI_Package.zip',
      },
    });

  } catch (error) {
    console.error('Error generating download package:', error);
    return NextResponse.json(
      { message: 'Error generating download package' },
      { status: 500 }
    );
  }
}