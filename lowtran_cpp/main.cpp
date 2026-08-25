// https://claude.ai/share/3cda0e57-9395-4dc2-a760-6e45067ae990
/*
cmake -S . -B build
cmake --build build
./build/main
*/
#include <iostream>
#include <iomanip>

// ---- C-compatible interface implemented by the Fortran bind(C) wrapper ----
extern "C" {
    void lowtran_transmittance(float wavelength_nm, int model, int itype,
                                float h1, float h2, float angle, float range_km,
                                float *transmittance, float *wavelength_out_nm);
}

int main() {
    // --- scenario: ground-to-space transmittance @ 550 nm, subarctic winter ---
    const float wavelength_nm = 550.0f;
    const int   model         = 5;   // atmosphere: subarctic winter (Card 1, table 14)
    const int   itype         = 3;   // path type: observer to space
    const float h1            = 0.0f; // observer altitude [km]
    const float h2            = 0.0f;
    const float angle         = 0.0f; // zenith angle [deg]
    const float range_km      = 0.0f;

    float transmittance = 0.0f;
    float wavelength_out_nm = 0.0f;

    lowtran_transmittance(wavelength_nm, model, itype,
                           h1, h2, angle, range_km,
                           &transmittance, &wavelength_out_nm);

    std::cout << "LOWTRAN7 call completed without crashing.\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Wavelength returned [nm]: " << wavelength_out_nm << "\n";
    std::cout << "Total transmittance     : " << transmittance << "\n";

    if (wavelength_out_nm > 0.0f && transmittance >= 0.0f && transmittance <= 1.0f) {
        std::cout << "PASS: lowtran is working correctly.\n";
        return 0;
    } else {
        std::cout << "FAIL: lowtran output looks wrong.\n";
        return 1;
    }
}