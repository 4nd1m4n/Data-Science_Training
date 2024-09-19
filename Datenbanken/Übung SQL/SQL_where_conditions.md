
# SQL WHERE CONDITIONS


## Introduction

Show tables in paintings database.

```sql
SHOW TABLES;
```

Contains "artist" table.


Show artist table top 10.

```sql
SELECT *
FROM artist
LIMIT 10;
```
(LIMIT used to not output to much - only 10 entries)


## WHERE

Show artist table top 10 but only if they have middle_names that aren't null ("IS NOT NULL" - where condition).

```sql
SELECT *
FROM artist
WHERE middle_names IS NOT NULL
LIMIT 10;
```

Unfortunately that did not work.
Maybe the "middle_names" column contains empty strings?


Second try with unequals empty string ('') where condition.

```sql
SELECT *
FROM artist
WHERE middle_names <> ''
LIMIT 10;
```

This works.


## WHERE & AND

Show artist table top 10 but only if they have middle_names and where born after 1850.

```sql
SELECT *
FROM artist
WHERE LENGTH(middle_names) > 0
    AND birth > 1850
LIMIT 10;
```

All entries have a middle name and are born after 1850 now.


## WHERE & OR

Show artist table top 10 but only if they have middle_names and where born after 1850 or before 1700.

```sql
SELECT *
FROM artist
WHERE LENGTH(middle_names) > 0
    AND birth > 1850
    OR birth < 1700
LIMIT 10;
```

Did not work as intended.
Now empty middle names come up again...?

Braces to the rescue!

```sql
SELECT *
FROM artist
WHERE LENGTH(middle_names) > 0
    AND (
        birth > 1850
        OR birth < 1700
    )
LIMIT 10;
```

Worked, now a few entries before 1700 birth year are present.


## WHERE & NOT

Show artist table top 10 but only if they have middle_names and are not of "American" nationality.

```sql
SELECT *
FROM artist
WHERE LENGTH(middle_names) > 0
    AND NOT nationality="American"
LIMIT 10;
```


## WHERE & LIKE

Show artist table top 10 but only if their first_name starts with an "An...".

```sql
SELECT *
FROM artist
WHERE first_name LIKE "An%"
LIMIT 10;
```


## WHERE & IN

Show artist table top 10 but only if their nationality is "German", "Swiss" or "Austrian".

```sql
SELECT *
FROM artist
WHERE nationality IN ("German", "Swiss", "Austrian")
LIMIT 10;
```


## WHERE & BETWEEN

Show artist table top 10 but only if their death is between 1900 - 1910.

```sql
SELECT *
FROM artist
WHERE death BETWEEN 1900 AND 1910
LIMIT 10;
```
