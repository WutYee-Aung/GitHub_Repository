FROM python:3.11.5-bookworm

RUN pip install --upgrade pip
RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install ipykernel
RUN pip3 install seaborn
RUN pip3 install scikit-learn
RUN pip3 install ppscore
RUN pip3 install setuptools
RUN pip3 install shap
RUN pip3 install Flask

# COPY ./Source_Code/a2_app /root/source_code/a2_app/

CMD tail -f /dev/null

# pip install ipykernel -U --user --force-reinstall
# np.__version__, pd.__version__, sns.__version__, matplotlib.__version__
# ('1.25.2', '2.0.3', '0.12.2', '3.7.2')