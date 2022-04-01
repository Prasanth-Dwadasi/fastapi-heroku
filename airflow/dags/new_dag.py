from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pendulum

def helloWorld():
    print("Hello World")



with DAG(dag_id="new_dag",
         start_date=pendulum.datetime(2022,3,31,tz='EST'),
         schedule_interval="@hourly",
         catchup=False) as dag:

        task1 = PythonOperator(
        task_id="hello_world",
        python_callable=helloWorld)

task1
