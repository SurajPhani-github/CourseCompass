def fix_indentation():
    with open('src/app/streamlit_app.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    in_block = False
    for i, line in enumerate(lines):
        if line.startswith('                with col_map:') and i > 1000:
            in_block = True
            
        if line.startswith('    # ── TAB 3: Master Course Catalog'):
            in_block = False
            
        if in_block and line.startswith('    '):
            lines[i] = line[4:]

    with open('src/app/streamlit_app.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)

if __name__ == "__main__":
    fix_indentation()
