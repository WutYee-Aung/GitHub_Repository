FROM mysql:latest

ENV MYSQL_ROOT_PASSWORD=root

COPY ./my.cnf /etc/mysql/my.cnf


# FROM mysql:latest

# ENV MYSQL_ROOT_PASSWORD=root

# COPY ./database_students.sql /docker-entrypoint-initdb.d/