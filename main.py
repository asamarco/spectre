import os
import re
from flask import Flask, send_file, request, jsonify
import numpy as np
from converter import get_cie_cmf, spectrum_to_xyz, xyz_to_rgb

app = Flask(__name__)

# Initialize the CIE color matching function interpolator
cie_cmf_func = get_cie_cmf()

def parse_spectrum_data(content):
    wavelengths = []
    spectrum_data = []
    lines = content.strip().split('\n')
    for line in lines:
        parts = re.split(r'[ ,\t]+', line.strip())
        if len(parts) >= 2:
            try:
                wavelengths.append(float(parts[0]))
                spectrum_data.append(float(parts[-1]))
            except ValueError:
                # Skip lines that cannot be converted to float
                pass
    return np.array(wavelengths), np.array(spectrum_data)

@app.route("/")
def index():
    return send_file('src/index.html')

@app.route("/convert", methods=['POST'])
def convert_spectrum():
    content = None
    # Check for file upload first
    if 'spectrum_file' in request.files and request.files['spectrum_file'].filename != '':
        file = request.files['spectrum_file']
        content = file.read().decode('utf-8')
    else:
        # If no file, check for JSON data (for pasted text)
        data = request.get_json(silent=True)
        if data and 'spectrum_text' in data:
            content = data['spectrum_text']

    if content is None:
        return jsonify({'error': 'No input data provided. Please upload a file or paste data.'}), 400

    try:
        wavelengths, spectrum_data = parse_spectrum_data(content)
        if len(wavelengths) == 0:
            return jsonify({'error': 'No valid data found in input'}), 400
    except Exception as e:
        return jsonify({'error': f'Error parsing data: {e}'}), 400

    # Convert the spectrum to an XYZ tristimulus value
    xyz = spectrum_to_xyz(wavelengths, spectrum_data, cie_cmf_func)
    
    # Convert XYZ to sRGB
    rgb = xyz_to_rgb(xyz)
    
    return jsonify({'rgb': rgb.tolist()})

def main():
    app.run(port=int(os.environ.get('PORT', 80)))

if __name__ == "__main__":
    main()
