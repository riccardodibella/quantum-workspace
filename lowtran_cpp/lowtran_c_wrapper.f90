! https://claude.ai/share/3cda0e57-9395-4dc2-a760-6e45067ae990
module lowtran_c_interface
  use iso_c_binding
  implicit none

contains

  subroutine lowtran_transmittance(wavelength_nm, model, itype, &
                                    h1, h2, angle, range_km, &
                                    transmittance, wavelength_out_nm) &
                                    bind(C, name="lowtran_transmittance")

    ! ---- C-visible interface: all scalars passed BY VALUE ----
    real(c_float),  value       :: wavelength_nm   ! input wavelength [nm]
    integer(c_int), value       :: model            ! atmosphere model (1-7)
    integer(c_int), value       :: itype            ! path type (1-3)
    real(c_float),  value       :: h1, h2           ! altitudes [km]
    real(c_float),  value       :: angle            ! zenith angle [deg]
    real(c_float),  value       :: range_km         ! slant range [km]
    real(c_float),  intent(out) :: transmittance    ! result
    real(c_float),  intent(out) :: wavelength_out_nm

    ! ---- internal LOWTRAN workspace (same as your test program) ----
    integer, parameter :: nwl = 1
    integer, parameter :: ml  = 1

    real :: v1, v2, dv
    real :: tx(nwl,63)
    real :: v(nwl), alam(nwl)
    real :: trace(nwl), unif(nwl)
    real :: suma(nwl)
    real :: irrad(nwl,3)
    real :: sumvv(nwl)

    integer :: iemsct, im, iseasn, ird1
    real :: zmdl(ml), p(ml), t(ml), wmol(12)

    v1 = 1.0e7 / wavelength_nm
    v2 = v1
    dv = 20.0

    iemsct = 0   ! transmittance-only
    im     = 0
    iseasn = 0
    ird1   = 0

    zmdl = 0.0
    p    = 0.0
    t    = 0.0
    wmol = 0.0

    call lwtrn7(.true., nwl, v1, v2, dv, &
                tx, v, alam, trace, unif, &
                suma, irrad, sumvv, &
                model, itype, iemsct, im, &
                iseasn, ml, ird1, &
                zmdl, p, t, wmol, &
                h1, h2, angle, range_km)

    transmittance     = tx(1,10)
    wavelength_out_nm = alam(1) * 1000.0

  end subroutine lowtran_transmittance

end module lowtran_c_interface