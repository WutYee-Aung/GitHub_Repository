create database student;
use student;

create table students(
    studentid int not null auto_increment,
    FirstName varchar(100) NOT NULL,
    Surname varchar(100) NOT NULL,
    PRIMARY KEY (studentid)
);

INSERT INTO students(FirstName,Surname)
VALUES("John","Anderson"),("Emma","Smith");