CREATE TABLE IF NOT EXISTS DETAILS (
    ID INT NOT NULL AUTO_INCREMENT,
    QUERY VARCHAR(1000) NOT NULL,
    RESPONSE TEXT NOT NULL,
    RATING VARCHAR(25),
    PRIMARY KEY(ID)
);

