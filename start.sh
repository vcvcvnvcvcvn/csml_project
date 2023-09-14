wget http://39.98.141.84:3838/iMarxTool/data.zip
unzip data.zip
cd model
python run_exp.py args_baseline.py
python run_exp.py args_ctm.py
python run_exp.py args_bt_baseline.py
python run_exp.py args_bt_ctm.py