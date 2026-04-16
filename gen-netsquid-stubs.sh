#!/bin/bash

# Ensure output directory exists
mkdir -p ./typings

# List of packages to stub
PACKAGES=("netsquid" "pydynaa")

for PKG in "${PACKAGES[@]}"; do
    # Get the path to the package dynamically
    PKG_PATH=$(python3 -c "import $PKG; print($PKG.__path__[0])" 2>/dev/null)
    
    if [ -z "$PKG_PATH" ]; then
        echo "Skipping $PKG: Not found in current environment."
        continue
    fi

    echo "Processing $PKG at $PKG_PATH..."

    # Loop through all compiled .so files
    for file in $(find "$PKG_PATH" -name "*.so"); do
        # 1. Match the package name in the path and keep everything after it
        # 2. Strip the extension
        # 3. Swap / for . 
        MODULE=$(echo "$file" | sed "s|.*/$PKG|$PKG|; s|\.cpython.*||; s|/|.|g")
        
        echo "  -> Stubbing: $MODULE"
        pybind11-stubgen "$MODULE" -o ./typings
    done

    # Add the secret handshake file for the IDE
    mkdir -p "./typings/$PKG"
    touch "./typings/$PKG/py.typed"
done