setup: requirements.txt
	pip install -r requirements.txt

run:
	python app.py

clean:
	rm -rf __pycache__