module def_mod

    implicit none
!    integer, parameter :: wp = selected_real_kind(6) ! single precision
    integer, parameter :: wp = selected_real_kind(14) ! double precision
    real(wp), parameter :: pi = 3.14159265358979_wp
    integer :: nb_nodes_org
    integer :: nb_edges_org
    integer :: max_nb_edges
    integer :: nb_div
    integer :: div_level
    integer, parameter :: ifinp = 10
    integer, parameter :: ifhyp = 11
    integer, parameter :: ifout = 12
    character(:), allocatable :: fninp
    character(:), allocatable :: fnhyp
    character(:), allocatable :: fnout
    integer , allocatable :: edges_org(:, :)
    real(wp), allocatable :: edge_weights_org(:)
    real(wp), allocatable :: adjacent_org(:, :)

end module def_mod

!======================================================================
!======================================================================

program rqaoa

    !$ use omp_lib
    use def_mod, only : wp, nb_nodes_org, nb_edges_org, edges_org, &
        & edge_weights_org, adjacent_org, max_nb_edges, ifout, fnout
    implicit none
    logical  :: flag
    integer  :: i, j, k, p, q, iseed, brute_force_threshold
    integer  :: nb_rand, clock, nb_nodes_new, nb_edges_new, i_cand, sgn_edge_energy
    integer  :: node_delete, node_keep, ios, nb_trees
    integer  :: time_begin, time_end, count_per_sec, count_max
    real(wp) :: edge_noise, gamma_opt, beta_opt, expectation, w, emax, cut, cut_best
    character(1) :: binary
    integer , allocatable :: seed(:)
    integer , allocatable :: edges_new(:, :)
    integer , allocatable :: bits_best(:)
    integer , allocatable :: parity(:, :)
    integer , allocatable :: tree_roots(:)
    real(wp), allocatable :: noise(:)
    real(wp), allocatable :: edge_weights_new(:)
    real(wp), allocatable :: edge_energy(:)
    real(wp), allocatable :: adjacent_new(:, :)

    call system_clock(time_begin, count_per_sec, count_max)
    
    call system_clock ( count=clock )
    call random_seed( size=nb_rand )
    allocate( seed(nb_rand) )
    seed(:) = clock
    call random_seed ( put=seed )
    
    call read_data (            &
        & brute_force_threshold, &
        & edge_noise             )
    
    nb_nodes_new = nb_nodes_org
    nb_edges_new = nb_edges_org
    
    allocate(                                      &
        & noise(nb_edges_org),                      &
        & edges_new(2, max_nb_edges),               &
        & edge_weights_new(max_nb_edges),           &
        & adjacent_org(nb_nodes_org, nb_nodes_org), &
        & adjacent_new(nb_nodes_org, nb_nodes_org), &
        & edge_energy(max_nb_edges),                &
        & parity(nb_nodes_org, nb_nodes_org),       &
        & tree_roots(nb_nodes_org),                 &
        & bits_best(nb_nodes_org)                   )

    parity(1:nb_nodes_org, 1:nb_nodes_org) = 0
        
    call random_number(noise)
    
    edges_new(1:2, 1:nb_edges_new) = edges_org(1:2, 1:nb_edges_new)
    edge_weights_new(1:nb_edges_new) = edge_weights_org(1:nb_edges_new) &
        & + edge_noise * (2_wp * noise - 1_wp)
    
    adjacent_org(1:nb_nodes_org, 1:nb_nodes_org) = 0_wp

    do i = 1, nb_edges_org

        p = edges_org(1, i)
        q = edges_org(2, i)
        w = edge_weights_org(i)
        adjacent_org(p, q) = w
        adjacent_org(q, p) = w
        
        w = edge_weights_new(i)
        adjacent_new(p, q) = w
        adjacent_new(q, p) = w

    end do

    open(unit=ifout, file=fnout, status='replace', action='write', iostat=ios )
    
    loop:do while (nb_nodes_new .gt. brute_force_threshold)
    
        call get_optimal_gamma_beta ( &
            & nb_nodes_new,           &
            & nb_edges_new,           &
            & edges_new,              &
            & edge_weights_new,       &
            & adjacent_new,           &
            & gamma_opt,              &
            & beta_opt                )
            
        call get_level1_qaoa_exp ( &
            & nb_edges_new,        &
            & gamma_opt,           &
            & beta_opt,            &
            & edges_new,           &
            & edge_weights_new,    &
            & adjacent_new,        &
            & expectation,         &
            & edge_energy          )
            
        emax = 0_wp
        i_cand = 1
        do i = 1, nb_edges_new
            if (abs(edge_energy(i)) .gt. emax) then
                emax = abs(edge_energy(i))
                i_cand = i
            end if
        end do
        
        node_delete = edges_new(1, i_cand)
        node_keep = edges_new(2, i_cand)
        
        if (edge_energy(i_cand) .gt. 0_wp) then
            sgn_edge_energy = 1
        else
            sgn_edge_energy = -1
        end if

        call get_new_graph( &
            & node_delete,      &
            & node_keep,        &
            & nb_edges_new,     &
            & edges_new,        &
            & edge_weights_new, &
            & adjacent_new,     &
            & sgn_edge_energy,  &
            & parity            )

        if (nb_edges_new .eq. 0) then
            exit loop
        end if
        
        nb_nodes_new = 0
        do i = 1, nb_nodes_org
            jj1:do j = 1, nb_nodes_org
                if (adjacent_new(i, j) .ne. 0) then
                    nb_nodes_new = nb_nodes_new + 1
                    exit jj1
                end if
            end do jj1
        end do

        write(*,*) '# nodes', nb_nodes_new
        
    end do loop

    call get_trees( &
        & parity,     &
        & nb_trees,   &
        & tree_roots  )

    call get_cut( &
        & nb_trees,   &
        & tree_roots, &
        & parity,     &
        & bits_best,  &
        & cut_best    )
    
    open(unit=ifout, file=fnout, status='replace', action='write', iostat=ios )
        
    write(*, '(A,f10.3)') 'cut (best) = ', cut_best
    write(ifout, '(10000i1)') bits_best
    write(ifout, *) cut_best
    
    call system_clock(time_end)
    write(ifout, *) dble(time_end - time_begin) / dble(count_per_sec)
    write(*, '(A,f15.3,A)') 'Wall time  = ', dble(time_end - time_begin) / dble(count_per_sec), ' [sec]'
    
    close( ifout )
        
    stop

end program

!======================================================================
!======================================================================

subroutine read_data ( &
    & brute_force_threshold, &
    & edge_noise             )

    use def_mod, only : wp, nb_nodes_org, nb_edges_org, edges_org, edge_weights_org, max_nb_edges, &
        & nb_div, div_level, ifinp, ifhyp, ifout, fninp, fnhyp, fnout
    implicit none
    integer , intent(out) :: brute_force_threshold
    real(wp), intent(out) :: edge_noise
    integer  :: ios, i, p, q, nb_edges = 0
    real(wp) :: w
    character(:), allocatable :: arg
    integer :: length, status
    intrinsic :: command_argument_count, get_command_argument
    
    do i = 0, command_argument_count()
        call get_command_argument(i, length=length, status=status)
        if (status .eq. 0) then
            allocate( character(length) :: arg )
            call get_command_argument(i, arg, status=status)
            if (status .eq. 0) then
                if (i .eq. 1) then
                    allocate( character(len(arg)) :: fninp )
                    fninp = arg
                    write(*,*) 'Input file name : ', fninp
                else if (i .eq. 2) then
                    allocate( character(len(arg)) :: fnhyp )
                    fnhyp = arg
                    write(*,*) 'Hyps file name : ', fnhyp
                else if (i .eq. 3) then
                    allocate( character(len(arg)) :: fnout )
                    fnout = arg
                    write(*,*) 'Hyps file name : ', fnout
                end if
            end if
            deallocate (arg)
        else
            write(*, *) 'Error', status, 'on argument', i
        end if
    end do
    
    open(unit=ifinp, file=fninp, status='old', action='read', iostat=ios )
    open(unit=ifhyp, file=fnhyp, status='old', action='read', iostat=ios )
    
    read(ifinp, *) nb_nodes_org, nb_edges_org
    
    allocate( edges_org(2, nb_edges_org), edge_weights_org(nb_edges_org) )
    
    do i = 1, nb_edges_org
        ! Node index must be started with 1 (node index 0 is prohibited)
        read(ifinp, *) p, q, w
        if ((p .eq. 0) .or. (q .eq. 0)) then
            write(*,*) 'Node numbers must start from 1.'
            stop
        end if
        edges_org(1, i) = p
        edges_org(2, i) = q
        edge_weights_org(i) = w
    end do
    
    nb_nodes_org = maxval(edges_org) - minval(edges_org) + 1
    max_nb_edges = nb_edges_org

    read(ifhyp, *) brute_force_threshold
    read(ifhyp, *) edge_noise
    read(ifhyp, *) nb_div
    read(ifhyp, *) div_level
    
    close( ifinp )
    close( ifhyp )
    
end subroutine

!======================================================================
!======================================================================

subroutine get_optimal_gamma_beta ( &
    & nb_nodes_new, &
    & nb_edges_new, &
    & edges,        &
    & edge_weights, &
    & adjacent,     &
    & gamma_opt,    &
    & beta_opt      )

    !$ use omp_lib
    use def_mod, only : wp, pi, nb_nodes_org, nb_edges_org, max_nb_edges, nb_div, div_level
    implicit none
    integer,  intent(in)  :: nb_nodes_new
    integer,  intent(in)  :: nb_edges_new
    integer,  intent(in)  :: edges(2, max_nb_edges)
    real(wp), intent(in)  :: edge_weights(max_nb_edges)
    real(wp), intent(in)  :: adjacent(nb_nodes_org, nb_nodes_org)
    real(wp), intent(out) :: gamma_opt
    real(wp), intent(out) :: beta_opt
    integer  :: i, level, ios
    real(wp) :: nb_div_inv, expectation, edge_energy(max_nb_edges), gamma, beta
    real(wp) :: exp_min, g_min, g_max, dg
    
    exp_min = 1e10_wp
    
    g_min = 0_wp
    g_max = pi
    dg = (g_max - g_min) / dble(nb_div - 1)

    do level = 1, div_level
    
        !$omp parallel private ( i, gamma, beta, expectation )
        !$omp do
        do i = 0, nb_div

            gamma = dble(i) * dg + g_min
            
            call get_optimal_beta( &
                & nb_nodes_new, &
                & nb_edges_new, &
                & gamma,        &
                & edges,        &
                & edge_weights, &
                & adjacent,     &
                & beta          )

            call get_level1_qaoa_exp ( &
                & nb_edges_new,        &
                & gamma,               &
                & beta,                &
                & edges,               &
                & edge_weights,        &
                & adjacent,            &
                & expectation,         &
                & edge_energy          )

            !$omp critical ( max )
            if (expectation .lt. exp_min) then

                exp_min = expectation
                gamma_opt = gamma
                beta_opt = beta

            end if
            !$omp end critical ( max )
            
        end do
        !$omp end do
        !$omp end parallel
        
        g_min = gamma_opt - dg
        g_max = gamma_opt + dg
        dg = (g_max - g_min) / dble(nb_div - 1)
        
    end do
    
end subroutine
    
!======================================================================
!======================================================================

subroutine get_optimal_beta( &
    & nb_nodes_new, &
    & nb_edges_new, &
    & gamma,        &
    & edges,        &
    & edge_weights, &
    & adjacent,     &
    & beta_opt )
    
    use def_mod, only : wp, max_nb_edges, pi, nb_nodes_org
    implicit none
    integer , intent(in) :: nb_nodes_new
    integer , intent(in) :: nb_edges_new
    real(wp), intent(in) :: adjacent(nb_nodes_org, nb_nodes_org)
    real(wp), intent(in) :: gamma
    integer , intent(in) :: edges(2, max_nb_edges)
    real(wp), intent(in) :: edge_weights(max_nb_edges)
    real(wp), intent(out) :: beta_opt
    integer :: i, j, j_count, p, q
    real(wp) :: gw, cos_gw, sin_gw, cpp, spp, gpp, gpm, gmp, gmm, w_p, w_q, cmm, smm, cmp, smp, cpm, spm
    real(wp) :: r11, r22, r33, r44, r14, r23, i12, i13, i24, i34, a, b, alpha
    complex(wp) :: pm, pp, mm, v025 = cmplx(0.25_wp, 0_wp), v05 = cmplx(0.5_wp, 0_wp)
    complex(wp) :: m0246(4, 8) = (0_wp, 0_wp)
    complex(wp) :: m1357(4, 8) = (0_wp, 0_wp)
    complex(wp) :: epls(8, 8) = (0_wp, 0_wp)
    complex(wp) :: emns(8, 8) = (0_wp, 0_wp)
    complex(wp) :: etapls(8, 8) = (0_wp, 0_wp)
    complex(wp) :: eta_sum(4, 4) = (0_wp, 0_wp)
    complex(wp) :: rho(8, 8) = (0_wp, 0_wp)
    complex(wp) :: eta(4, 4, nb_edges_new)
    integer :: neighbor(nb_nodes_new, nb_edges_new)
    
    m0246(1, 1) = cmplx(1_wp, 0_wp)
    m0246(2, 3) = cmplx(1_wp, 0_wp)
    m0246(3, 5) = cmplx(1_wp, 0_wp)
    m0246(4, 7) = cmplx(1_wp, 0_wp)
    m1357(1, 2) = cmplx(1_wp, 0_wp)
    m1357(2, 4) = cmplx(1_wp, 0_wp)
    m1357(3, 6) = cmplx(1_wp, 0_wp)
    m1357(4, 8) = cmplx(1_wp, 0_wp)
    
    neighbor(1:nb_nodes_new, 1:nb_edges_new) = 0
    do i = 1, nb_edges_new
        p = edges(1, i)
        q = edges(2, i)
        j_count = 0
        do j = 1, nb_nodes_new
            if ((adjacent(p, j) .ne. 0_wp) .or. (adjacent(q, j) .ne. 0_wp)) then
                if ((j .ne. p) .and. (j .ne. q)) then
                    j_count = j_count + 1
                    neighbor(j_count, i) = j
                end if
            end if
        end do
    end do

    do i = 1, nb_edges_new

        gw = gamma * edge_weights(i)
        pp = v025 * cmplx(cos(2_wp * gw),  sin(2_wp * gw))
        mm = v025 * cmplx(cos(2_wp * gw), -sin(2_wp * gw))
        eta(1, 1, i) = v025
        eta(1, 2, i) = pp
        eta(1, 3, i) = pp
        eta(1, 4, i) = v025
        eta(2, 1, i) = mm
        eta(2, 2, i) = v025
        eta(2, 3, i) = v025
        eta(2, 4, i) = mm
        eta(3, 1, i) = mm
        eta(3, 2, i) = v025
        eta(3, 3, i) = v025
        eta(3, 4, i) = mm
        eta(4, 1, i) = v025
        eta(4, 2, i) = pp
        eta(4, 3, i) = pp
        eta(4, 4, i) = v025
    
    end do
    
    do i = 1, nb_edges_new
    
        p = edges(1, i)
        q = edges(2, i)
        j_count = 1
        
        serch_neighbor:do while (.true.)
        
            if (neighbor(j_count, i) .eq. 0) then
            
                exit serch_neighbor
                
            else

                j = neighbor(j_count, i)
                w_p = adjacent(p, j)
                w_q = adjacent(q, j)
                gpp = gamma * (  w_p + w_q)
                gpm = gamma * (  w_p - w_q)
                gmp = gamma * (- w_p + w_q)
                gmm = gamma * (- w_p - w_q)
                cpp = cos(gpp)
                cmm = cos(gmm)
                cpm = cos(gpm)
                cmp = cos(gmp)
                spp = sin(gpp)
                smm = sin(gmm)
                spm = sin(gpm)
                smp = sin(gmp)
                
                epls(1, 1) = cmplx(cpp, spp)
                epls(2, 2) = cmplx(cmm, smm)
                epls(3, 3) = cmplx(cpm, spm)
                epls(4, 4) = cmplx(cmp, smp)
                epls(5, 5) = cmplx(cmp, smp)
                epls(6, 6) = cmplx(cpm, spm)
                epls(7, 7) = cmplx(cmm, smm)
                epls(8, 8) = cmplx(cpp, spp)
                
                emns(1, 1) = cmplx(cmm, smm)
                emns(2, 2) = cmplx(cpp, spp)
                emns(3, 3) = cmplx(cmp, smp)
                emns(4, 4) = cmplx(cpm, spm)
                emns(5, 5) = cmplx(cpm, spm)
                emns(6, 6) = cmplx(cmp, smp)
                emns(7, 7) = cmplx(cpp, spp)
                emns(8, 8) = cmplx(cmm, smm)
                
                etapls(1, 1) = v05 * eta(1, 1, i)
                etapls(1, 2) = v05 * eta(1, 1, i)
                etapls(2, 1) = v05 * eta(1, 1, i)
                etapls(2, 2) = v05 * eta(1, 1, i)
                etapls(1, 3) = v05 * eta(1, 2, i)
                etapls(1, 4) = v05 * eta(1, 2, i)
                etapls(2, 3) = v05 * eta(1, 2, i)
                etapls(2, 4) = v05 * eta(1, 2, i)
                etapls(1, 5) = v05 * eta(1, 3, i)
                etapls(1, 6) = v05 * eta(1, 3, i)
                etapls(2, 5) = v05 * eta(1, 3, i)
                etapls(2, 6) = v05 * eta(1, 3, i)
                etapls(1, 7) = v05 * eta(1, 4, i)
                etapls(1, 8) = v05 * eta(1, 4, i)
                etapls(2, 7) = v05 * eta(1, 4, i)
                etapls(2, 8) = v05 * eta(1, 4, i)

                etapls(3, 1) = v05 * eta(2, 1, i)
                etapls(3, 2) = v05 * eta(2, 1, i)
                etapls(4, 1) = v05 * eta(2, 1, i)
                etapls(4, 2) = v05 * eta(2, 1, i)
                etapls(3, 3) = v05 * eta(2, 2, i)
                etapls(3, 4) = v05 * eta(2, 2, i)
                etapls(4, 3) = v05 * eta(2, 2, i)
                etapls(4, 4) = v05 * eta(2, 2, i)
                etapls(3, 5) = v05 * eta(2, 3, i)
                etapls(3, 6) = v05 * eta(2, 3, i)
                etapls(4, 5) = v05 * eta(2, 3, i)
                etapls(4, 6) = v05 * eta(2, 3, i)
                etapls(3, 7) = v05 * eta(2, 4, i)
                etapls(3, 8) = v05 * eta(2, 4, i)
                etapls(4, 7) = v05 * eta(2, 4, i)
                etapls(4, 8) = v05 * eta(2, 4, i)

                etapls(5, 1) = v05 * eta(3, 1, i)
                etapls(5, 2) = v05 * eta(3, 1, i)
                etapls(6, 1) = v05 * eta(3, 1, i)
                etapls(6, 2) = v05 * eta(3, 1, i)
                etapls(5, 3) = v05 * eta(3, 2, i)
                etapls(5, 4) = v05 * eta(3, 2, i)
                etapls(6, 3) = v05 * eta(3, 2, i)
                etapls(6, 4) = v05 * eta(3, 2, i)
                etapls(5, 5) = v05 * eta(3, 3, i)
                etapls(5, 6) = v05 * eta(3, 3, i)
                etapls(6, 5) = v05 * eta(3, 3, i)
                etapls(6, 6) = v05 * eta(3, 3, i)
                etapls(5, 7) = v05 * eta(3, 4, i)
                etapls(5, 8) = v05 * eta(3, 4, i)
                etapls(6, 7) = v05 * eta(3, 4, i)
                etapls(6, 8) = v05 * eta(3, 4, i)

                etapls(7, 1) = v05 * eta(4, 1, i)
                etapls(7, 2) = v05 * eta(4, 1, i)
                etapls(8, 1) = v05 * eta(4, 1, i)
                etapls(8, 2) = v05 * eta(4, 1, i)
                etapls(7, 3) = v05 * eta(4, 2, i)
                etapls(7, 4) = v05 * eta(4, 2, i)
                etapls(8, 3) = v05 * eta(4, 2, i)
                etapls(8, 4) = v05 * eta(4, 2, i)
                etapls(7, 5) = v05 * eta(4, 3, i)
                etapls(7, 6) = v05 * eta(4, 3, i)
                etapls(8, 5) = v05 * eta(4, 3, i)
                etapls(8, 6) = v05 * eta(4, 3, i)
                etapls(7, 7) = v05 * eta(4, 4, i)
                etapls(7, 8) = v05 * eta(4, 4, i)
                etapls(8, 7) = v05 * eta(4, 4, i)
                etapls(8, 8) = v05 * eta(4, 4, i)
                
                rho = matmul(matmul(epls, etapls), emns)
                eta(1:4, 1:4, i) = matmul(matmul(m0246, rho), transpose(m0246)) + matmul(matmul(m1357, rho), transpose(m1357))
                
            end if

            j_count = j_count + 1
            
        end do serch_neighbor
        
    end do
        
    eta_sum(1:4, 1:4) = cmplx(0_wp, 0_wp)
    do i = 1, nb_edges_new
        eta_sum(1:4, 1:4) = eta_sum(1:4, 1:4) + edge_weights(i) * eta(:, :, i)
    end do
    
    r11 = real(eta_sum(1, 1))
    r22 = real(eta_sum(2, 2))
    r33 = real(eta_sum(3, 3))
    r44 = real(eta_sum(4, 4))
    r14 = real(eta_sum(1, 4))
    r23 = real(eta_sum(2, 3))
    i12 = imag(eta_sum(1, 2))
    i13 = imag(eta_sum(1, 3))
    i24 = imag(eta_sum(2, 4))
    i34 = imag(eta_sum(3, 4))
    
    a = -i12 - i13 + i24 + i34
    b = (r11 - r22 - r33 + r44) / 2_wp + r14 - r23
    
    if (a .eq. 0_wp) then
        alpha = 0_wp
    else
        alpha = atan2(b, a)
    end if

    beta_opt = - alpha / 4_wp + pi / 8_wp
    
    do while (beta_opt < 0_wp)
        beta_opt = beta_opt + 0.5_wp * pi
    end do
    
    do while (beta_opt > 0.5_wp * pi)
        beta_opt = beta_opt - 0.5_wp * pi
    end do
        
end subroutine

!======================================================================
!======================================================================

subroutine get_level1_qaoa_exp ( &
    & nb_edges,     &
    & gamma,        &
    & beta,         &
    & edges,        &
    & edge_weights, &
    & adjacent,     &
    & expectation,  &
    & edge_energy   )

    use def_mod, only : wp, nb_nodes_org, nb_edges_org, max_nb_edges
    implicit none
    integer , intent(in) :: nb_edges
    real(wp), intent(in) :: gamma
    real(wp), intent(in) :: beta
    integer , intent(in) :: edges(2, max_nb_edges)
    real(wp), intent(in) :: edge_weights(max_nb_edges)
    real(wp), intent(in) :: adjacent(nb_nodes_org, nb_nodes_org)
    real(wp), intent(out) :: expectation
    real(wp), intent(out) :: edge_energy(max_nb_edges)
    integer :: i, p, uu, vv, u, v
    real(wp) :: c, s, w, cos1, cos2, cos3, cos4, gamma_x_2

    c = cos(2_wp * beta)
    s = sin(2_wp * beta)
    gamma_x_2 = 2_wp * gamma
    expectation = 0_wp

    do i = 1, nb_edges
    
        uu = edges(1, i)
        vv = edges(2, i)
        u = min(uu, vv)
        v = max(uu, vv)
        
        cos1 = 1_wp
        cos2 = 1_wp
        cos3 = 1_wp
        cos4 = 1_wp
        
        do p = 1, nb_nodes_org

            if ((p .ne. u) .and. (p .ne. v)) then
            
                cos1 = cos1 * (cos(gamma_x_2 * adjacent(u, p) - gamma_x_2 * adjacent(v, p)))
                cos2 = cos2 * (cos(gamma_x_2 * adjacent(u, p) + gamma_x_2 * adjacent(v, p)))
                cos3 = cos3 * cos(gamma_x_2 * adjacent(u, p))
                cos4 = cos4 * cos(gamma_x_2 * adjacent(v, p))
                
            end if
            
        end do
        
        edge_energy(i) = 0.5_wp * s * s * (cos1 - cos2) &
            & + c * s * sin(gamma_x_2 * adjacent(u, v)) * (cos3 + cos4)
            
        expectation = expectation + adjacent(u, v) * edge_energy(i)
    
    end do

end subroutine

!======================================================================
!======================================================================

subroutine get_new_graph ( &
    & node_delete,      &
    & node_keep,        &
    & nb_edges_new,     &
    & edges_new,        &
    & edge_weights_new, &
    & adjacent,         &
    & sgn_edge_energy,  &
    & parity            )

    use def_mod, only : wp, nb_nodes_org, nb_edges_org, max_nb_edges
    implicit none
    integer , intent(in) :: node_delete
    integer , intent(in) :: node_keep
    integer , intent(in) :: sgn_edge_energy
    integer , intent(out) :: nb_edges_new
    integer , intent(out) :: edges_new(2, max_nb_edges)
    real(wp), intent(out) :: edge_weights_new(max_nb_edges)
    real(wp), intent(inout) :: adjacent(nb_nodes_org, nb_nodes_org)
    integer , intent(inout) :: parity(nb_nodes_org, nb_nodes_org)
    integer  :: i, j
    real(wp) :: w
    real(wp) :: adjacent_new(nb_nodes_org, nb_nodes_org)
    integer , allocatable :: foo(:, :)
    real(wp), allocatable :: bar(:)

    adjacent_new = 0_wp
    
    do i = 1, nb_nodes_org
        if (i .ne. node_delete) then
            do j = i + 1, nb_nodes_org
                w = adjacent(i, j)
                adjacent_new(i, j) = adjacent_new(i, j) + w
                adjacent_new(j, i) = adjacent_new(j, i) + w
            end do
        else
            do j = 1, nb_nodes_org
                if (j .ne. node_keep) then
                    w = dble(sgn_edge_energy) * adjacent(node_delete, j)
                    adjacent_new(node_keep, j) = adjacent_new(node_keep, j) + w
                    adjacent_new(j, node_keep) = adjacent_new(j, node_keep) + w
                end if
            end do
        end if
    end do
    
    adjacent = adjacent_new
    adjacent(node_delete, 1:nb_nodes_org) = 0_wp
    adjacent(1:nb_nodes_org, node_delete) = 0_wp
    
    parity(node_delete, node_keep) = sgn_edge_energy
    parity(node_keep, node_delete) = sgn_edge_energy
    
    allocate( foo(2, nb_edges_new), bar(nb_edges_new) )

    nb_edges_new = 0
    do i = 1, nb_nodes_org
        do j = i + 1, nb_nodes_org
            if (adjacent(i, j) .ne. 0_wp) then
                nb_edges_new = nb_edges_new + 1
                edges_new(1, nb_edges_new) = i
                edges_new(2, nb_edges_new) = j
                edge_weights_new(nb_edges_new) = adjacent(i, j)
            end if
        end do
    end do

end subroutine

!======================================================================
!======================================================================

subroutine get_trees( &
    & parity,     &
    & nb_trees,   &
    & tree_roots  )

    use def_mod, only : wp, nb_nodes_org
    implicit none
    integer, intent(in)  :: parity(nb_nodes_org, nb_nodes_org)
    integer, intent(out) :: nb_trees
    integer, intent(out) :: tree_roots(nb_nodes_org)
    integer :: i, j, p, queue_header, i_queue, tree, tree_header, i_count
    integer :: queue(nb_nodes_org)
    integer :: tree_index(nb_nodes_org)
        
    tree_index(1:nb_nodes_org) = 0
    tree_roots(1:nb_nodes_org) = 0
    tree = 0
    tree_header = 1
    i_count = 0

    tree_loop:do while (.true.)
        
        tree = tree + 1
        tree_index(tree_header) = tree
        i_count = i_count + 1
        tree_roots(i_count) = tree_header
        
        queue(1:nb_nodes_org) = 0
        queue(1) = tree_header
        queue_header = 1
        i_queue = 1
        
        do while (i_queue .le. queue_header)
            i = queue(i_queue)
            do j = 1, nb_nodes_org
                p = parity(i, j)
                if ((p .ne. 0) .and. (tree_index(j) .eq. 0)) then
                    tree_index(j) = tree
                    queue_header = queue_header + 1
                    queue(queue_header) = j
                end if
            end do
            i_queue = i_queue + 1
        end do

        if (tree_header .eq. nb_nodes_org) then
            exit tree_loop
        end if

        tree_header_loop:do j = tree_header + 1, nb_nodes_org
            
            if (tree_index(j) .eq. 0) then
                tree_header = j
                exit tree_header_loop
            end if
            
            if (j .eq. nb_nodes_org) then
                tree_header = j
                exit tree_loop
            end if
            
        end do tree_header_loop
        
    end do tree_loop
    
    nb_trees = tree
    
end subroutine

!======================================================================
!======================================================================

subroutine get_cut( &
    & nb_trees,   &
    & tree_roots, &
    & parity,     &
    & bits_best,  &
    & cut_best    )

    use def_mod, only : wp, nb_nodes_org, nb_edges_org, edges_org, edge_weights_org
    implicit none
    integer,  intent(in)  :: nb_trees
    integer,  intent(in)  :: tree_roots(nb_nodes_org)
    integer,  intent(in)  :: parity(nb_nodes_org, nb_nodes_org)
    integer,  intent(out) :: bits_best(nb_nodes_org)
    real(wp), intent(out) :: cut_best
    character(:), allocatable :: bin
    integer  :: i, j, k, p, q, queue_header, i_queue
    integer  :: filled_parity(nb_nodes_org, nb_nodes_org)
    integer  :: queue(nb_nodes_org)
    integer  :: bits(nb_nodes_org)
    real(wp) :: cut

    allocate(character(len=nb_trees) :: bin)
    
    if (nb_trees .ge. 1) then
    
        cut_best = -1e10
        bin = repeat('0', nb_trees - 1)

        do k = 0, 2**(nb_trees - 1) - 1

            do i = 1, nb_trees
                if (ibits(k, nb_trees-i-1, 1) == 1) then
                    bin(i:i) = '1'
                else
                    bin(i:i) = '0'
                end if
            end do
            
            filled_parity(1:nb_nodes_org, 1:nb_nodes_org) = parity(1:nb_nodes_org, 1:nb_nodes_org)

            do i = 1, nb_trees - 1
                read(bin(i:i), *) p
                filled_parity(tree_roots(i), tree_roots(i+1)) = 2 * p - 1
                filled_parity(tree_roots(i+1), tree_roots(i)) = 2 * p - 1
            end do

            bits(1:nb_nodes_org) = 2
            queue(1:nb_nodes_org) = 0
            queue(1) = 1
            queue_header = 1
            bits(1) = 0
            i_queue = 1
            do while (i_queue .le. queue_header)
                i = queue(i_queue)
                do j = 1, nb_nodes_org
                    p = filled_parity(i, j)
                    if ((p .ne. 0) .and. (bits(j) .eq. 2)) then
                        if (p .eq. 1) then
                            bits(j) = bits(i)
                        else
                            bits(j) = 1 - bits(i)
                        end if
                        queue_header = queue_header + 1
                        queue(queue_header) = j
                    end if
                end do
                i_queue = i_queue + 1
            end do

            cut = 0_wp
            do i = 1, nb_edges_org
                p = edges_org(1, i)
                q = edges_org(2, i)
                if (bits(p) .ne. bits(q)) then
                    cut = cut + edge_weights_org(i)
                end if
            end do
            write(*, '(A,100I1)') 'bits = ', bits
            write(*, '(A,f10.3)') 'cut = ', cut

            if (cut .gt. cut_best) then
                cut_best = cut
                bits_best = bits
            end if

        end do
        
    else
    
        bits(1:nb_nodes_org) = 2
        queue(1:nb_nodes_org) = 0
        queue(1) = 1
        queue_header = 1
        bits(1) = 0
        i_queue = 1
        do while (i_queue .le. queue_header)
            i = queue(i_queue)
            do j = 1, nb_nodes_org
                p = parity(i, j)
                if ((p .ne. 0) .and. (bits(j) .eq. 2)) then
                    if (p .eq. 1) then
                        bits(j) = bits(i)
                    else
                        bits(j) = 1 - bits(i)
                    end if
                    queue_header = queue_header + 1
                    queue(queue_header) = j
                end if
            end do
            i_queue = i_queue + 1
        end do
        
        cut = 0_wp
        do i = 1, nb_edges_org
            p = edges_org(1, i)
            q = edges_org(2, i)
            if (bits(p) .ne. bits(q)) then
                cut = cut + edge_weights_org(i)
            end if
        end do
        
        cut_best = cut
        
    end if

end subroutine