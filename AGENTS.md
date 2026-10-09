# Repository working agreements

## Finishing a task

Unless the user explicitly requests otherwise, a task is complete only after
its validated changes are committed, integrated, pushed, and installed:

1. Run the checks appropriate to the changes and commit the completed task's
   changes. Preserve unrelated work, including changes in other worktrees.
2. When working in a Git worktree or task branch, merge the task into the local
   default branch (`master` in this repository). Use its existing checkout,
   preserve unrelated uncommitted changes, and resolve routine conflicts within
   the task's scope.
3. Push the integrated default branch to `origin`. Verify that the remote branch
   contains the completed changes. This is standing authorization to commit,
   merge locally, and push; no separate confirmation is needed.
4. On macOS, build the client from the final integrated source using
   `scripts/build_macos_client.py` and the configured packaging environment.
   Install the entire resulting `Kinect 3D Scanner.app` bundle into
   `~/Applications/Kinect 3D Scanner.app`, replacing the previous installed
   build. A build left only in a worktree's `dist` folder is unfinished work.
5. Run the installed app's `--check` from outside the checkout and verify that
   the installed executable matches the checked build. If the old app is
   running, restart it with the installed build when it is idle. Preserve active
   captures and unsaved work; do not discard them to restart. For UI changes,
   verify the changed behavior in the installed app.
6. Report the pushed commit and installed app path. If a concrete blocker
   prevents pushing, building, installing, or safely restarting, report the
   blocker and remaining steps instead of declaring completion.

Installing the Mac client does not update a separately running reconstruction
server. State clearly when that server still needs to pull the changes and
restart.
