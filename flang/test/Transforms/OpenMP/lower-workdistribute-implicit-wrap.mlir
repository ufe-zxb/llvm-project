// Implicit-workdistribute wraps each candidate `fir.do_loop unordered`.
// host: in `omp.teams { omp.workdistribute { ... } }` (lowered to the
// teams/parallel/distribute/wsloop nest). device: the same, inside an
// `omp.target` with an implicit `omp.map.info` per live-in. none/default: no-op.

// RUN: fir-opt --lower-workdistribute='implicit=host'   %s | FileCheck %s --check-prefixes=BOTH,HOST
// RUN: fir-opt --lower-workdistribute='implicit=device' %s | FileCheck %s --check-prefixes=BOTH,DEVICE
// RUN: fir-opt --lower-workdistribute='implicit=none'   %s | FileCheck %s --check-prefixes=BOTH,NOOP
// RUN: fir-opt --lower-workdistribute                   %s | FileCheck %s --check-prefixes=BOTH,NOOP

// BOTH-LABEL:   func.func @candidate(
// HOST:           omp.teams {
// HOST:             omp.parallel {
// HOST:               omp.distribute {
// HOST:                 omp.wsloop {
// HOST:                   omp.loop_nest (%{{.*}}) : index = (%{{.*}}) to (%{{.*}}) inclusive step (%{{.*}}) {
// HOST:                     arith.constant 0 : index
// HOST:                     fir.store
// HOST:                     omp.yield
// HOST:                   }
// HOST:                 } {omp.composite}
// HOST:               } {omp.composite}
// HOST:               omp.terminator
// HOST:             } {omp.composite}
// HOST:             omp.terminator
// HOST:           }
// The device kernel is SPMD and {omp.combined} (so the verifier can reach the
// captured loop_nest). Its bounds are host-evaluated, so the loop_nest reads
// the host_eval block args directly. Only `addr` is a mapped live-in here:
// bounds that feed no in-body op are host_eval-only, no extra map.info.
// DEVICE:         omp.target kernel_type(spmd)
// DEVICE-SAME:      host_eval(%{{[^[:space:]]+}} -> %[[C_LB:[^ ,]+]], %{{[^[:space:]]+}} -> %[[C_UB:[^ ,]+]], %{{[^[:space:]]+}} -> %[[C_ST:[^ ,]+]] : index, index, index)
// DEVICE:           omp.loop_nest (%{{.*}}) : index = (%[[C_LB]]) to (%[[C_UB]]) inclusive step (%[[C_ST]]) {
// DEVICE:         } {omp.combined}
// NOOP-NOT:       omp.teams
// NOOP-NOT:       omp.workdistribute
// NOOP:           fir.do_loop %{{[^ ]+}} = %{{[^ ]+}} to %{{[^ ]+}} step %{{[^ ]+}} unordered
// BOTH:           return
func.func @candidate(%lb : index, %ub : index, %step : index,
                     %addr : !fir.ref<index>) {
  fir.do_loop %iv = %lb to %ub step %step unordered {
    %zero = arith.constant 0 : index
    fir.store %zero to %addr : !fir.ref<index>
  }
  return
}

// Ordered loops (regular Fortran DO) are not wrapped in any mode.
// BOTH-LABEL:   func.func @ordered_skipped(
// BOTH-NOT:       omp.teams
// BOTH-NOT:       omp.workdistribute
// BOTH-NOT:       omp.target
// BOTH:           fir.do_loop
// BOTH:           return
func.func @ordered_skipped(%lb : index, %ub : index, %step : index,
                           %addr : !fir.ref<index>) {
  fir.do_loop %iv = %lb to %ub step %step {
    %zero = arith.constant 0 : index
    fir.store %zero to %addr : !fir.ref<index>
  }
  return
}

// A loop already inside an explicit workdistribute is honored as-is in every
// mode: the implicit wrap is skipped, and the rest of the pass lowers the
// existing `omp.teams { omp.workdistribute { ... } }` normally.
// BOTH-LABEL:   func.func @explicit_passthrough(
// BOTH:           omp.teams {
// BOTH:             omp.parallel {
// BOTH:               omp.distribute {
// BOTH:                 omp.wsloop {
// BOTH:                   omp.loop_nest
// BOTH-NOT:       omp.teams
// BOTH-NOT:       omp.target
func.func @explicit_passthrough(%lb : index, %ub : index, %step : index,
                                %addr : !fir.ref<index>) {
  omp.teams {
    omp.workdistribute {
      fir.do_loop %iv = %lb to %ub step %step unordered {
        %zero = arith.constant 0 : index
        fir.store %zero to %addr : !fir.ref<index>
      }
      omp.terminator
    }
    omp.terminator
  }
  return
}

// Loops already inside an `omp.parallel` are user-managed and not wrapped
// in any mode.
// BOTH-LABEL:   func.func @parallel_skipped(
// BOTH:           omp.parallel {
// BOTH-NOT:       omp.teams
// BOTH-NOT:       omp.target
// BOTH:           fir.do_loop
// BOTH:           omp.terminator
func.func @parallel_skipped(%lb : index, %ub : index, %step : index,
                            %addr : !fir.ref<index>) {
  omp.parallel {
    fir.do_loop %iv = %lb to %ub step %step unordered {
      %zero = arith.constant 0 : index
      fir.store %zero to %addr : !fir.ref<index>
    }
    omp.terminator
  }
  return
}

// Only the outermost unordered loop is wrapped (host/device). The inner
// unordered loop rides along unchanged inside the body. In none/default mode
// both loops are left as bare `fir.do_loop`s.
// BOTH-LABEL:   func.func @nested_only_outer(
// HOST:           omp.teams {
// HOST:             omp.parallel {
// HOST:               omp.distribute {
// HOST:                 omp.wsloop {
// HOST:                   omp.loop_nest
// HOST:                     fir.do_loop {{.*}} unordered
// The inner loop reuses the outer bounds. host_eval args may only feed omp ops,
// so the bounds are ALSO mapped and loaded on the device: the outer loop_nest
// consumes the host_eval args, while the inner fir.do_loop consumes the loaded
// device copies.
// DEVICE:         omp.target kernel_type(spmd)
// DEVICE-SAME:      host_eval(%{{[^[:space:]]+}} -> %[[N_LB:[^ ,]+]], %{{[^[:space:]]+}} -> %[[N_UB:[^ ,]+]], %{{[^[:space:]]+}} -> %[[N_ST:[^ ,]+]] : index, index, index)
// DEVICE-SAME:      map_entries(
// DEVICE:           %[[N_LB_LD:.*]] = fir.load %{{.*}} : !fir.ref<index>
// DEVICE:           %[[N_UB_LD:.*]] = fir.load %{{.*}} : !fir.ref<index>
// DEVICE:           %[[N_ST_LD:.*]] = fir.load %{{.*}} : !fir.ref<index>
// DEVICE:           omp.loop_nest (%{{.*}}) : index = (%[[N_LB]]) to (%[[N_UB]]) inclusive step (%[[N_ST]]) {
// DEVICE:             fir.do_loop %{{.*}} = %[[N_LB_LD]] to %[[N_UB_LD]] step %[[N_ST_LD]] unordered
// DEVICE:         } {omp.combined}
// NOOP-NOT:       omp.teams
// NOOP-NOT:       omp.workdistribute
// NOOP:           fir.do_loop {{.*}} unordered
// NOOP:             fir.do_loop {{.*}} unordered
func.func @nested_only_outer(%lb : index, %ub : index, %step : index,
                             %addr : !fir.ref<index>) {
  fir.do_loop %iv = %lb to %ub step %step unordered {
    fir.do_loop %iv2 = %lb to %ub step %step unordered {
      %zero = arith.constant 0 : index
      fir.store %zero to %addr : !fir.ref<index>
    }
  }
  return
}
