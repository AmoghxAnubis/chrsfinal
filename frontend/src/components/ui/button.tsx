import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  type?: "button" | "submit" | "reset"; // Updated type prop to match expected types
  variant?: 'default' | 'outline';
  className?: string;
  onClick?: () => void;
  disabled?: boolean;
  children: React.ReactNode;
}

const Button: React.FC<ButtonProps> = ({ variant = 'default', className, onClick, disabled, children }) => {
  const baseStyles = 'py-2 px-4 rounded transition-all duration-300';
  const variantStyles = variant === 'outline' 
    ? 'border border-purple-300 text-purple-300 hover:bg-purple-300 hover:text-white' 
    : 'bg-purple-600 text-white hover:bg-purple-700';

  return (
    <button 
      className={`${baseStyles} ${variantStyles} ${className}`} 
      onClick={onClick} 
      disabled={disabled}
    >
      {children}
    </button>
  );
};

export default Button;
