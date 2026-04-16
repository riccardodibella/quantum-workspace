#!/bin/bash

# Get the path to the netsquid folder dynamically
NS_PATH=$(python3 -c "import netsquid; print(netsquid.__path__[0])")

# Ensure the output directory exists
mkdir -p ./typings

# Loop through all compiled .so files
for file in $(find "$NS_PATH" -name "*.so"); do
    # Strip path prefix, strip extension, and swap / for .
    MODULE=$(echo "$file" | sed "s|.*/netsquid|netsquid|; s|\.cpython.*||; s|/|.|g")
    
    echo "Stubs for: $MODULE"
    # Output specifically to the typings folder
    pybind11-stubgen "$MODULE" -o ./typings
done

# Touch py.typed to make it official for the IDE
touch ./typings/netsquid/py.typed 2>/dev/null