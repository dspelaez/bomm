! -*- coding: utf-8 -*-
! vim:fenc=utf-8
!
! ===================================================================
! core.f90
! Copyright (C) 2018 Daniel Santiago <dpelaez@gmail.com>
!
! Distributed under terms of the GNU/GPL license.
!
!
!    To compile this module use:
!        >> f2py -c core.f90 -m core
!
! ===================================================================


!     position_and_phase {{{
! ===================================================================
      subroutine position_and_phase(W, x, y, XX, Dphi, &
                                    nfrqs, ntime, npoints, neqs)
!
!     This subroutine computes the wavenumber pair (kx, ky) for a
!     given wavelts coefficients and positions. This subroutine is
!     though to be part of the Wavelet Directional Method proposed 
!     by Donelan et al. (1996)
    

!     varible declaration    
      implicit none
      complex*8, intent(in) :: W(nfrqs, ntime, npoints)
      real*8, intent(in)    :: x(ntime, npoints), y(ntime, npoints)
      integer, intent(in)   :: nfrqs, ntime, npoints, neqs
!
      integer :: i, j, k, n, ij
      real*8  :: phase(nfrqs, ntime, npoints)
!
      real*8, intent(out) :: XX(ntime, neqs, 2), Dphi(nfrqs, ntime, neqs)
!

!     compute and phase from the wavelet coefficients
      phase = atan2(aimag(W), real(W))

      ! loop for each unique pair of points
      !    i,j ---> move along points
      !    k   ---> move along time
      !    n   ---> move along frequencies
      !    ij  ---> move along number of pairs
      ij = 1
      do i = 1, npoints
        !
        do j = i+1, npoints
          !
          ! distances between pairs for each time
          do k = 1, ntime
            XX(k,ij,1) = x(k,j) - x(k,i)
            XX(k,ij,2) = y(k,j) - y(k,i)
            !
            ! difference of phases
            do n = 1, nfrqs
              Dphi(n,k,ij) = phase(n,k,i) - phase(n,k,j)
            end do
          !
          end do
          !
          ! acumulate counter
          ij = ij + 1
        end do
      end do
      
      end subroutine
!     }}}

!     compute_wavenumbers {{{
! ===================================================================
      subroutine compute_wavenumber(XX, Dphi, kx, ky, &
                                    nfrqs, ntime, neqs)

!     apply least squares method to estimate the wavenumber
      
!     variable declaration
      implicit none
!
      real*8, intent(in)  :: XX(ntime, neqs, 2), Dphi(nfrqs, ntime, neqs)
      integer, intent(in) :: nfrqs, ntime, neqs
      real*8, intent(out) :: kx(nfrqs, ntime), ky(nfrqs, ntime)
!
      integer :: i, j, m, n
      real*8 :: detinv
      real*8 :: XTX(2,2), XTX_inv(2,2), XTDphi(2), kk(2)

      ! loop for each time
      do j = 1, ntime
        !
        ! loop for each frequency
        do i = 1, nfrqs
          !
          ! --- least square estimation of vector kk=(kx, ky) ---
          !     the LSR of k is given by:
          !       kk^LS = (X^T X)^-1 X^T Dphi
          !
          ! first term
          XTX(:,:) = matmul(transpose(XX(j,:,:)), XX(j,:,:)) 
          !
          ! inverse of the first term
          detinv = 1.d0 / (XTX(1,1)*XTX(2,2) - XTX(1,2)*XTX(2,1))
          do n = 1, 2
            do m = 1, 2
              XTX_inv(m,n) = ((-1)**(m+n)) * detinv * XTX(n,m)
            end do
          end do
          !
          ! second term
          XTDphi(:) = matmul(transpose(XX(j,:,:)), Dphi(i,j,:))
          !
          ! compute wavenumber components
          kk(:) = matmul(XTX_inv, XTDphi)
          kx(i,j) = kk(1)
          ky(i,j) = kk(2)
          !
        end do
      end do

      end subroutine
!     }}}

!     directional_spreading {{{
! ===================================================================

      subroutine directional_spreading(W, kx, ky, D, &
                                       nfrqs, ntime, npoints)

!     This function computes the directional spreading function. It is
!     based on finding the 
      
!     variable declaration
      implicit none
!
      complex*8, intent(in) :: W(nfrqs, ntime, npoints)
      real*8, intent(in)    :: kx(nfrqs, ntime), ky(nfrqs, ntime)
      integer, intent(in)   :: nfrqs, ntime, npoints
!
      integer :: i, j, t, ix(ntime)
      real*8 :: kappa(nfrqs, ntime), theta(nfrqs, ntime)
      real*8 :: power(nfrqs, ntime)
      real*8 :: theta_degrees(nfrqs, ntime), weight, m0(nfrqs)
      real*8, parameter :: pi = 3.14159265359
!
      real*8, intent(out) :: D(360, nfrqs)


!     compute power of each pair frequency-time
      power = sum(abs(W)**2, 3) / npoints

!     compute magnitude and direction of the wavenumber vector
      kappa = sqrt(kx**2 + ky**2)
      theta = atan2(ky, kx)

!     round angles to a resolution of 1 degree and correct angle to be
!     measured counterclockwise from east and wrapped from 0 to 360
      theta_degrees = modulo(nint(theta * 180./pi), 360)

!     loop for each frequcency
      do j = 1, nfrqs
        ! 
        ! loop for each direction
        do i = 0, 359
          !
          ! loop for each time
          do t = 1, ntime
            if (theta_degrees(j,t) == i) then
              ix(t) = 1
            else
              ix(t) = 0
            end if
          end do
          !
          ! comute weight
          if (sum(ix) == 0) then
            weight = 0.
          else
            weight = sum(power(j,ix)) / ntime
          end if
          !
          ! compute the directional spreading function
          D(i+1,j) = sum(ix) * weight
          !
        end do
      end do

!     normalize to satisfy int(D) = 1 for each direction
      m0 = sum(D, 1) * pi/180
      do i = 1, 360
        D(i,:) = D(i,:) / m0(:)
      end do

      end subroutine
!     }}}

!     compute_fdir_spectrum {{{
! ===================================================================

      subroutine compute_fdir_spectrum(W, kx, ky, E,        &
                                       nfrqs, ntime, npoints)

!     This function computes the wavenumber-direction wave spectrum
!     averaging the occurences of each case in the power matrix.
      
!     variable declaration
      implicit none
!
      complex*8, intent(in) :: W(nfrqs, ntime, npoints)
      real*8, intent(in)    :: kx(nfrqs, ntime), ky(nfrqs, ntime)
      integer, intent(in)   :: nfrqs, ntime, npoints
!
      integer :: f, d, t
      real*8 :: power(nfrqs, ntime)
      real*8 :: theta(nfrqs, ntime)
      real*8, parameter :: pi = 3.14159265359
      real*8, intent(out) :: E(360, nfrqs)


!     compute power of each pair frequency-time
      power = sum(abs(W)**2, 3) / npoints

!     compute magnitude and direction of the wavenumber vector
      theta = modulo(atan2(ky, kx) * 180./pi, 360.)

!     f ---> iterate over frequencies
!     t ---> iterate over time

      ! initilize matrix with zeros
      E(:, :) = 0.

!     loop for each wavenumber
      do t = 1, ntime
        !
        ! loop for each direction
        do f = 1, nfrqs
          !
          ! map direction to its corresponding index
          d = nint(theta(f,t) + 1)
          !
          ! compute the frequency-wavenumber spectrum
          E(d,f) = E(d,f) + power(f,t)
          !  
        end do
      end do

      end subroutine
!     }}}

!     compute_kdir_spectrum {{{
! ===================================================================

      subroutine compute_kdir_spectrum(W, kx, ky, E, wnum,         &
                                       nfrqs, ntime, nwnum, npoints)

!     This function computes the wavenumber-direction wave spectrum
!     averaging the occurences of each case in the power matrix.
      
!     variable declaration
      implicit none
!
      complex*8, intent(in) :: W(nfrqs, ntime, npoints)
      real*8, intent(in)    :: kx(nfrqs, ntime), ky(nfrqs, ntime)
      real*8, intent(in)    :: wnum(nwnum)
      integer, intent(in)   :: nfrqs, ntime, npoints, nwnum
!
      integer :: f, d, t, k
      real*8 :: power(nfrqs, ntime)
      real*8 :: kappa(nfrqs, ntime)
      real*8 :: theta(nfrqs, ntime)
      real*8 :: kmin, kmax, dk
      real*8, parameter :: pi = 3.14159265359
      real*8, intent(out) :: E(360, nwnum)


!     compute power of each pair frequency-time
      power = sum(abs(W)**2, 3) / npoints

!     compute magnitude and direction of the wavenumber vector
      kappa = sqrt(kx**2 + ky**2)
      theta = modulo(atan2(ky, kx) * 180./pi, 360.)

!     f ---> iterate over frequencies
!     k ---> iterate over wavenumbers
!     t ---> iterate over time

!     compute delta k and mininum k
      dk =  wnum(2) - wnum(1)
      kmin = wnum(1)
      kmax = wnum(nwnum)

      ! initilize matrix with zeros
      E(:, :) = 0.

!     loop for each wavenumber
      do t = 1, ntime
        !
        ! loop for each direction
        do f = 1, nfrqs
          !
          ! compute the frequency-wavenumber spectrum
          if (kappa(f,t) .le. kmax) then
            !
            ! map direction to its corresponding index
            d = nint(theta(f,t) + 1)
            k = nint((kappa(f,t) - kmin) / dk + 1)
            !
            E(d,k) = E(d,k) + power(f,t)
            !
          end if
          !  
        end do
      end do

      end subroutine
!     }}}

!     compute_kxky_spectrum {{{
! ===================================================================

      subroutine compute_kxky_spectrum(W, kx, ky, kxbin, kybin, E, &
                              nfrqs, nkxbin, nkybin, ntime, npoints)

!     This function computes the kx-ky wave spectrum summing the
!     energy of each case in the power matrix.
      
!     variable declaration
      implicit none
!
      complex*8, intent(in) :: W(nfrqs, ntime, npoints)
      real*8, intent(in)    :: kx(nfrqs, ntime), ky(nfrqs, ntime)
      real*8, intent(in)    :: kxbin(nkxbin), kybin(nkybin)
      integer, intent(in)   :: nfrqs, nkxbin, nkybin, ntime, npoints
!
      integer :: f, t, jkx, jky
      real*8 :: power(nfrqs, ntime), dkx, dky
      real*8, parameter :: pi = 3.14159265359
      real*8, intent(out) :: E(nkybin, nkxbin)


!     compute power of each pair frequency-time
      power = sum(abs(W)**2, 3) / npoints

!     arrays must be linearily spaced
      dkx =  kxbin(2) - kxbin(1)
      dky =  kybin(2) - kybin(1)

!     f ---> iterate over frequencies
!     t ---> iterate over time
!     jkx -> iterate over wavenumbers x
!     jkx -> iterate over wavenumbers y
      
      ! initilize matrix with zeros
      E(:,:) = 0.

!     loop for each wavenumber
      do t = 1, ntime
        !
        ! loop for each direction
        do f = 1, nfrqs
          !
          ! TODO: discard indices outside the range of wavenumbers
          if (abs(kx(f,t)) .le. kxbin(nkxbin) .and. &
            & abs(ky(f,t)) .le. kybin(nkybin)) then
            !
            jkx = nint((kx(f,t) - kxbin(1)) / dkx + 1)
            jky = nint((ky(f,t) - kxbin(1)) / dky + 1)
            !
            ! compute the wavenumber spectrum
            E(jky,jkx) = E(jky,jkx) + power(f,t)
            !
          end if
          !  
        end do
      end do

      end subroutine
!     }}}

!     random phase 1d {{{
! ===================================================================

      subroutine randomphase_oned(time, frqs, E, eta, ntime, nfrqs)

      ! return the surface elevation asociated with the given
      ! frequency spectrum

      implicit none

      real*8, intent(in)   :: time(ntime), frqs(nfrqs), E(nfrqs)
      integer, intent(in)  :: nfrqs, ntime
      !
      real*8               :: ampl, omega, phase, df
      integer              :: j
      real*8, parameter    :: pi = 3.141592653589793
      !
      real*8, intent(out)  :: eta(ntime)
      !
      ! delta frequency
      df = frqs(2) - frqs(1)
      !
      ! initialize surface elevation
      eta(:) = 0.
      !
      do j = 1, nfrqs
          !
          ampl = sqrt(2. * E(j) * df)
          omega = 2.*pi*frqs(j)
          phase = 2.*pi*rand(0) - pi
          !
          ! compute one dimensioanl surface elevation
          eta = eta + ampl * sin(omega * time + phase)
      !
      end do
      
      end subroutine
!     }}}

!     random phase 2d {{{
! ===================================================================

      subroutine randomphase_twod(time, frqs, dirs, E, eta, &
                                  xgrd, ygrd, nx, ny,       &
                                  ntime, nfrqs, ndirs)

      ! return the surface elevation asociated with the given
      ! direction-frequency spectrum

      implicit none

      real*8, intent(in)   :: time(ntime), E(ndirs, nfrqs)
      real*8, intent(in)   :: frqs(nfrqs), dirs(ndirs)
      real*8, intent(in)   :: xgrd(ny, nx), ygrd(ny, nx)
      integer, intent(in)  :: nfrqs, ndirs, ntime, nx, ny
      !
      real*8               :: ampl, omega, phase, df, dd
      real*8               :: kphase(ny, nx), kappa, kx, ky
      integer              :: i, j, t
      real*8, parameter    :: g = 9.80
      real*8, parameter    :: pi = 3.141592653589793
      !
      real*8, intent(out)  :: eta(ntime, ny, nx)
      !
      ! delta frequency and delta theta
      df = frqs(2) - frqs(1)
      dd = (dirs(2) - dirs(1)) * pi/180.
      !
      ! initialize surface elevation
      eta(:,:,:) = 0.
      !
      do i = 1, ndirs
        !
        do j = 1, nfrqs
          !
          ampl = sqrt(2. * E(i, j) * df * dd)
          omega = 2.*pi*frqs(j)
          kappa = omega ** 2 / g
          kx = kappa * cos(dirs(i) * pi/180.)
          ky = kappa * sin(dirs(i) * pi/180.)
          phase = 2.*pi*rand(0) - pi
          !
          ! compute two dimensioanl surface elevation
          do t = 1, ntime
            !
            kphase = kx*xgrd + ky*ygrd - omega*time(t) + phase
            eta(t,:,:) = eta(t,:,:) + ampl*cos(kphase)
            !
          end do
          !
        end do
        !
      end do

    end subroutine
!     }}}


! --- end of file ---

