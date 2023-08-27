FROM python:3.11.4-bookworm

RUN pip3 install numpy==1.25.2
RUN pip3 install pandas==2.0.3
RUN pip3 install ipykernel==6.25.1
RUN pip3 install seaborn==0.12.2
RUN pip3 install scikit-learn==1.3.0
RUN pip3 install ppscore==1.3.0
RUN pip3 install setuptools==68.1.2
RUN pip3 install shap==0.42.1
RUN pip3 install Flask

CMD tail -f /dev/null

# pip install ipykernel -U --user --force-reinstall
# np.__version__, pd.__version__, sns.__version__, matplotlib.__version__
# ('1.25.2', '2.0.3', '0.12.2', '3.7.2')