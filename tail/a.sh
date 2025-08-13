nohup python3 a.py input.csv > app.log 2>&1 &

tail -f app.log





ps aux | grep your_script.py
kill 12345