---
name: finder-file-ops
apps: com.apple.finder
triggers: file, folder, rename, move, copy, duplicate, delete, trash, reveal, organize
requires_scripting: true
---
For Finder file operations, one approved AppleScript beats a click sequence,
and its output is your verification.

## Script verbs
- New folder: tell application "Finder" to make new folder at (path to desktop folder) with properties {name:"X"}
- Rename: tell application "Finder" to set name of (POSIX file "/path/to/file" as alias) to "newname.ext"
- Move: tell application "Finder" to move (POSIX file "/path/a" as alias) to (POSIX file "/path/dir" as alias)
- Duplicate: tell application "Finder" to duplicate (POSIX file "/path/a" as alias)
- Reveal: tell application "Finder" to reveal (POSIX file "/path/a" as alias)

## Deleting
Use the moveToTrash action with the absolute path — never the Finder GUI and
never a delete keystroke. Permanent deletion does not exist.

## Verification
The script result line in your step result is the ground truth; if it errors,
fix the path or fall back to GUI steps for that one operation.
