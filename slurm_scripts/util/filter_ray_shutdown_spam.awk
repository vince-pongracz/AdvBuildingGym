#!/usr/bin/env -S awk -f
#
# filter_ray_shutdown_spam.awk
#
# Strips the harmless EnvRunner.__del__ + sigterm_handler traceback blocks
# that Ray prints to stderr when env-runner actors are SIGTERM'd at the end
# of tuner.fit().  These blocks look like (one per worker pid):
#
#   (SingleAgentEnvRunner pid=N) Exception ignored in: <function EnvRunner.__del__ ...>
#   (SingleAgentEnvRunner pid=N) Traceback (most recent call last):
#   (SingleAgentEnvRunner pid=N)   File ".../ray/rllib/env/env_runner.py", line 205, in __del__
#   (SingleAgentEnvRunner pid=N)     def __del__(self) -> None:
#   (SingleAgentEnvRunner pid=N)
#   (SingleAgentEnvRunner pid=N)   File ".../ray/_private/worker.py", line 1041, in sigterm_handler
#   (SingleAgentEnvRunner pid=N)     raise_sys_exit_with_custom_error_message(
#   (SingleAgentEnvRunner pid=N)   File "python/ray/_raylet.pyx", line 677, in ray._raylet.raise_sys_exit_with_custom_error_message
#   (SingleAgentEnvRunner pid=N) SystemExit: 1
#
# State machine, keyed by worker pid prefix:
#   Idle      --  start_re  -->  Suppress
#   Suppress  --  end_re    -->  Idle   (the end line is also dropped)
#
# SLURM stderr can interleave concurrent writes from the driver and worker
# processes (observed: a real "INFO - main - Optimized metric ..." line
# concatenated onto a "... in sigterm_handler" frame).  When a suppressed
# line also carries a real driver log timestamp, the trailing real content
# is preserved.

BEGIN {
    pid_re   = "\\((Single|Multi)AgentEnvRunner pid=[0-9]+\\)"
    start_re = "Exception ignored in: <function EnvRunner\\.__del__"
    end_re   = "SystemExit:"
    # ISO timestamp at the start of a salvaged log line, e.g. "2026-05-11 13:55:54,475".
    log_re   = "[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}"
}

{
    pid = ""
    if (match($0, pid_re)) {
        pid = substr($0, RSTART, RLENGTH)
    }

    salvaged = ""
    if (match($0, log_re)) {
        salvaged = substr($0, RSTART)
    }

    if (pid != "" && (pid in suppress)) {
        if (match($0, end_re)) {
            delete suppress[pid]
        }
        if (salvaged != "") {
            print salvaged
            fflush()
        }
        next
    }

    if (pid != "" && match($0, start_re)) {
        suppress[pid] = 1
        if (salvaged != "") {
            print salvaged
            fflush()
        }
        next
    }

    print
    fflush()
}
