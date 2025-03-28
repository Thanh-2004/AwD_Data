# AwD_Data

## User instruction:

1. **Clone the repository:**
```
git clone 
cd AwD_Data/AwD_data_v2
```

2. **Install dependencies:**

- Create a virtual environment:
```
python3 -m venv AWD
```
_Make sure your current Python version is 3.x_

- Activate the environment
```
source AWD/bin/activate
```
or
```
AWD\Scripts\activate
```

for macOS/Linux and Window OS respectively


- Install libraries, packages and dependancies:

```
pip install -r requirements.txt
```

3. **Usage:**

- For real-time running: change the port corresponding to your computer
- For demo running: change code at the top of streamlit_test.py 
from 
```from collectData_streamlit import start_thread ``` 
to
```from collectData_streamlit_demo import start_thread ```

- To run:
```
streamlit run streamlit_test.py
```

