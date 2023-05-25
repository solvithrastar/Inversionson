from optson.optimizer import Optimizer
from optson.methods import AdamUpdate, TRUpdate, SteepestDescentUpdate
from optson.stopping_criterion import BasicStoppingCriterion
from optson.monitor import BasicMonitor
from inversionson.problem import Problem
from inversionson.utils import mesh_to_vector
from inversionson.project import Project
from inversionson.problem import InversionsonUpdatePrecondtioner
from inversionson.batch_manager import InversionsonBatchManager


def optson_run_config(project: Project):
    """
    This function will be extracted and become user configurable.
    """

    sc = BasicStoppingCriterion(
        tolerance=1e-100, divergence_tolerance=1e100, max_iterations=99999
    )
    monitor = BasicMonitor(step=1)

    ibm = InversionsonBatchManager(
        project=project,
        batch_size=project.config.inversion.initial_batch_size,
        use_overlapping_batches=True,
    )

    # Dynamic mini-batches with L-BFGS (uncomment if you want this)
    # problem = Problem(project=project, smooth_gradients=True)  # for Everything else
    # st_upd = SteepestDescentUpdate(initial=0.03, step_size_as_percentage=True)
    # update = TRUpdate(fallback=st_upd, verbose=True)
    # opt = Optimizer(
    #     problem=problem,
    #     update=update,
    #     stopping_criterion=sc,
    #     monitor=monitor,
    #     batch_manager=ibm,
    # )
    ###

    # Adam optimization (comment if using the above)
    problem = Problem(project=project, smooth_gradients=False)  # For Adam
    update_precond = InversionsonUpdatePrecondtioner(project=project)
    ibm.use_overlapping_batches = False
    ad_upd = AdamUpdate(
        alpha=0.005, epsilon=0.1, relative_epsilon=True, preconditioner=update_precond
    )
    ####

    # For operation without mini-batches, simply leave the batch_manager keyword empty.
    opt = Optimizer(
        problem=problem,
        update=ad_upd,
        stopping_criterion=sc,
        monitor=monitor,
        # batch_manager=ibm,
    )

    opt.iterate(
        x0=mesh_to_vector(
            project.lasif.master_mesh,
            params_to_invert=project.config.inversion.inversion_parameters,
        )
    )
