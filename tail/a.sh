nohup python3 a.py final_output.csv > app.log 2>&1 &

tail -f app.log





ps aux | grep your_script.py
kill 12345