subroutine run_lowtran_c(model, itype, iemsct, im, ihaze, h1, h2, angle, v1, v2, dv, out_arr) &
    bind(C, name="run_lowtran_c")
  use iso_c_binding
  implicit none

  integer(c_int), intent(in) :: model, itype, iemsct, im, ihaze
  real(c_float), intent(in) :: h1, h2, angle, v1, v2, dv
  real(c_float), intent(out) :: out_arr(*)

  interface
    subroutine trans(model, itype, iemsct, im, ihaze, h1, h2, angle, v1, v2, dv, out_arr)
      import :: c_int, c_float
      integer(c_int), intent(in) :: model, itype, iemsct, im, ihaze
      real(c_float), intent(in) :: h1, h2, angle, v1, v2, dv
      real(c_float), intent(out) :: out_arr(*)
    end subroutine trans
  end interface

  call trans(model, itype, iemsct, im, ihaze, h1, h2, angle, v1, v2, dv, out_arr)
end subroutine run_lowtran_c