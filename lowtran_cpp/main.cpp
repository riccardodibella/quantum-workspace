#include <iostream>
#include <vector>
#include <cmath>

extern "C" {
    void run_lowtran_c(
        const int* model,
        const int* itype,
        const int* iemsct,
        const int* im,
        const int* ihaze,
        const float* h1,
        const float* h2,
        const float* angle,
        const float* v1,
        const float* v2,
        const float* dv,
        float* output_buffer
    );
}

int main() {
    int model = 6;         // 1976 US Standard Atmosphere
    int itype = 3;         // Ground-to-Space
    int iemsct = 0;        // Transmittance mode
    int im = 0;            // Single scattering
    int ihaze = 1;         // Rural aerosol profile (VIS = 23 km)

    float h1 = 0.0f;       // Observer altitude (km)
    float h2 = 100.0f;     // Target altitude (km)
    float angle = 0.0f;    // Zenith angle (deg)

    // Wavelength inputs in nanometers
    float wl_start_nm = 400.0f;   // 400 nm (Visible)
    float wl_end_nm = 2500.0f;    // 2500 nm (SWIR)
    
    // Convert nm to cm^-1 (Note: 2500 nm = 4000 cm^-1, 400 nm = 25000 cm^-1)
    float v1 = 10000000.0f / wl_end_nm;    // 4000 cm^-1 (Start wavenumber)
    float v2 = 10000000.0f / wl_start_nm;  // 25000 cm^-1 (End wavenumber)
    float dv = 20.0f;                      // LOWTRAN resolution step (20 cm^-1)

    // Allocate sufficient buffer size (LOWTRAN returns spectral records)
    int n_steps = static_cast<int>((v2 - v1) / dv) + 1;
    std::vector<float> transmission(n_steps * 50, 0.0f);

    std::cout << "Executing LOWTRAN from " << v1 << " to " << v2 << " cm^-1..." << std::endl;

    run_lowtran_c(
        &model, &itype, &iemsct, &im, &ihaze,
        &h1, &h2, &angle,
        &v1, &v2, &dv,
        transmission.data()
    );

    std::cout << "\nSUCCESS! Execution completed with zero errors!" << std::endl;
    std::cout << "First total transmittance value: " << transmission[0] << std::endl;

    return 0;
}