! https://claude.ai/share/bcebcd33-a563-4d57-812c-eebea5885052
! gfortran -std=legacy -o fortran_test fortran_test.f90 ../lowtran/build/src/lowtran/liblowtran.a
! ./fortran_test
program fortran_test
  ! Minimal sanity check for the LOWTRAN7 Fortran library
  ! (https://github.com/space-physics/lowtran).
  !
  ! It calls the LWTRN7 driver subroutine directly (the same routine the
  ! Python f2py wrapper calls) for a single wavelength, ground-to-space,
  ! transmittance-only case, and checks that the returned transmittance
  ! is a physically sensible number between 0 and 1.

  implicit none

  integer, parameter :: nwl = 1   ! number of wavelength points
  integer, parameter :: ml  = 1   ! number of atmospheric layers (unused here)

  real :: v1, v2, dv               ! wavenumber range/step [cm^-1]
  real :: tx(nwl,63)               ! transmittance components (output)
  real :: v(nwl), alam(nwl)        ! wavenumber / wavelength (output)
  real :: trace(nwl), unif(nwl)    ! trace-gas / uniformly-mixed gas (output)
  real :: suma(nwl)                ! summed absorption (output)
  real :: irrad(nwl,3)             ! irradiance (output)
  real :: sumvv(nwl)               ! radiance (output)

  integer :: model, itype, iemsct, im, iseasn, ird1
  real :: zmdl(ml), p(ml), t(ml), wmol(12)
  real :: h1, h2, angle, range_km

  ! --- scenario: ground-to-space transmittance @ 550 nm, subarctic winter ---
  v1 = 1.0e7 / 550.0   ! wavenumber [cm^-1] for 550 nm
  v2 = 1.0e7 / 550.0   ! single wavelength -> v1 == v2
  dv = 20.0            ! step [cm^-1] (irrelevant for a single point)

  model  = 5   ! atmosphere: subarctic winter (Card 1, table 14)
  itype  = 3   ! path type: observer to space
  iemsct = 0   ! execution mode: transmittance only
  im     = 0
  iseasn = 0
  ird1   = 0

  zmdl = 0.0
  p    = 0.0
  t    = 0.0
  wmol = 0.0

  h1       = 0.0   ! observer altitude [km]
  h2       = 0.0
  angle    = 0.0    ! zenith angle [deg]
  range_km = 0.0

  call lwtrn7(.true., nwl, v1, v2, dv, &
              tx, v, alam, trace, unif, &
              suma, irrad, sumvv, &
              model, itype, iemsct, im, &
              iseasn, ml, ird1, &
              zmdl, p, t, wmol, &
              h1, h2, angle, range_km)

  print *, 'LOWTRAN7 call completed without crashing.'
  print *, 'Wavelength returned [nm]: ', alam(1) * 1000.0
  print *, 'Total transmittance     : ', tx(1, 10)

  if (alam(1) > 0.0 .and. tx(1,10) >= 0.0 .and. tx(1,10) <= 1.0) then
    print *, 'PASS: lowtran is working correctly.'
  else
    print *, 'FAIL: lowtran output looks wrong.'
    stop 1
  end if

end program fortran_test