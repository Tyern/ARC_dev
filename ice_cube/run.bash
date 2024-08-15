rm -rf absres-c-files
cp -r input/arc-solution-source-files-by-icecuber ./absres-c-files
cd absres-c-files; make -j > /dev/null
make -j count_tasks
python3 safe_run.py

cp absres-c-files/submission_part.csv old_submission.csv