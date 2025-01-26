import os

def list_python_files():
    python_files = []
    script_name = os.path.basename(__file__)  # Get the name of this script
    
    # Walk through all directories
    for root, dirs, files in os.walk('.'):
        # Skip virtual environment and git directories
        dirs[:] = [d for d in dirs if d not in {'.venv', 'venv', '.git'}]
            
        # Add all Python files in current directory
        for file in files:
            if file.endswith('.py') and file != script_name:  # Exclude this script
                full_path = os.path.join(root, file)
                # Convert to relative path and normalize
                rel_path = os.path.normpath(full_path)
                python_files.append(rel_path)
    
    # Sort files for consistent output
    python_files.sort()
    
    # Write to markdown file
    with open('python_files.md', 'w') as f:
        f.write('# Python Files in Repository\n\n')
        for file in python_files:
            f.write(f'## {file}\n\n')
            try:
                with open(file, 'r', encoding='utf-8') as source_file:
                    content = source_file.read()
                    f.write('```python\n')
                    f.write(content)
                    if not content.endswith('\n'):
                        f.write('\n')
                    f.write('```\n\n')
            except Exception as e:
                f.write(f'Error reading file: {str(e)}\n\n')

if __name__ == '__main__':
    list_python_files() 