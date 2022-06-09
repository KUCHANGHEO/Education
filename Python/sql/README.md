# SQL_Python

## 00. pymysql connect

- mysql.Connect 명령어로 직접 연결

## 01. read_db_config

- read_db_config로 config.ini에 적힌 데이터 베이스 연결

## 02. mysql fetchone

- cursor()에 연결
- fetchone으로 row 하나씩 읽기
- while문으로 row print

## 03. mysql fetchall

- fetchall로 row 한꺼번에 읽기
- for문으로 rows print

## 04. mysql fetchmany

- fetchmany로 있는만큼 row 출력하기

## 04.1 mysql fetchall like

- cursor.execute()에 query를 넣어 조건탐색

## 05. mysql insert

- cursor.excute()로 insert

## 05-1. mysql insert many

- 여러 row를 insert

## 06. mysql update

- cursor.excute()에 update query문 넣어 업데이트
- update이후 commit

## 07. mysql delete

- cursor.excute()에 delete query문 넣어 삭제

## 08. mysql Update_BLOB

## 09. mysql read_BLOB

## 10. mysql Stored Procedures

## 11. mysql pandas

- pd.read_sql_query()를 이용하여 DB에서 데이터 받아 데이터 프레임 만들기
