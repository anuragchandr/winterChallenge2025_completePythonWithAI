# 📦 Products Database — SQL Learning Guide

## 🔧 Create Table

```sql
CREATE TABLE products(
  product_id SERIAL PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  sku_code CHAR(8) UNIQUE NOT NULL,
  price NUMERIC(10,2) CHECK (price > 0),
  stock_quantity INT DEFAULT 0 CHECK (stock_quantity >= 0),
  is_available BOOLEAN DEFAULT TRUE,
  category TEXT NOT NULL,
  adden_on DATE DEFAULT CURRENT_DATE,
  last_update TIMESTAMP DEFAULT NOW()
);
```

## 📟 Insert Sample Data

```sql
INSERT INTO products (name, sku_code, price , stock_quantity, is_available, category)
VALUES
('Wireless Mouse', 'WM123456', 699.99, 50, TRUE, 'Electronics'),
('Bluetooth Speaker', 'BS234567', 1499.00, 30, TRUE, 'Electronics'),
('Laptop Stand', 'LS345678', 799.50, 20, TRUE, 'Accessories'),
('USB-C Hub', 'UC456789', 1299.99, 15, TRUE, 'Accessories'),
('Notebook', 'NB567890', 99.99, 100, TRUE, 'Stationery'),
('Pen Set', 'PS678901', 199.00, 200, TRUE, 'Stationery'),
('Coffee Mug', 'CM789012', 299.00, 75, TRUE, 'Home & Kitchen'),
('LED Desk Lamp', 'DL890123', 899.00, 40, TRUE, 'Home & Kitchen'),
('Yoga Mat', 'YM901234', 499.00, 25, TRUE, 'Fitness'),
('Water Bottle', 'WB012345', 349.00, 60, TRUE, 'Fitness');
```

---

## ✏️ Basic Clause Practice Questions

### Q1. Show the name and price of all products.

```sql
SELECT name, price FROM products;
```

### Q2. Show all products where the category is 'Electronics'.

```sql
SELECT * FROM products WHERE category = 'Electronics';
```

### Q3. Group products by category. Show each category once.

```sql
SELECT category FROM products GROUP BY category;
```

### Q4. Show categories that have more than 1 product.

```sql
SELECT category, COUNT(*) FROM products
GROUP BY category
HAVING COUNT(*) > 1;
```

### Q5. Show all products sorted by price in ascending order.

```sql
SELECT * FROM products ORDER BY price ASC;
```

### Q6. Show only the first 3 products from the table.

```sql
SELECT * FROM products LIMIT 3;
```

### Q7. Show product name as "Item\_Name" and price as "Item\_Price".

```sql
SELECT name AS Item_Name, price AS Item_Price FROM products;
```

### Q8. Show all the unique categories from the products table.

```sql
SELECT DISTINCT category FROM products;
```

---

## 🎓 Test 2 Questions

### Q1. Display the name and price of the cheapest product in the entire table.

```sql
SELECT name, price FROM products
WHERE price = (SELECT MIN(price) FROM products);
```

### Q2. Find the average price of products that belong to 'Home & Kitchen' or 'Fitness'.

```sql
SELECT category, AVG(price) AS avg_price
FROM products
WHERE category IN ('Home & Kitchen', 'Fitness')
GROUP BY category;
```

### Q3. Show product names and stock quantity where product is available, stock > 50, and price != 299.

```sql
SELECT name, stock_quantity FROM products
WHERE is_available = TRUE
AND stock_quantity > 50
AND price != 299.00;
```

### Q4. Find the most expensive product in each category.

```sql
SELECT category, MAX(price) AS max_price
FROM products
GROUP BY category;
```

### Q5. Show all unique categories in uppercase, sorted in descending order.

```sql
SELECT DISTINCT UPPER(category) AS category_upper
FROM products
ORDER BY category_upper DESC;
```

---

## 👩‍🎓 One-to-One Relationship: Students & Profiles

### Tables

```sql
CREATE TABLE students (
  student_id SERIAL PRIMARY KEY,
  name VARCHAR(100) NOT NULL
);

CREATE TABLE student_profiles (
  student_id INT PRIMARY KEY,
  address TEXT,
  age INT,
  phone VARCHAR(15)
);
```

### Insert Data

```sql
INSERT INTO students (name)
VALUES
('Akarsh Vyas'), ('Simran Mehta'), ('Rohan Gupta');

INSERT INTO student_profiles (student_id, address, age, phone)
VALUES
(1, 'Delhi, India', 22, '9999999999'),
(2, 'Mumbai, India', 21, '8888888888'),
(3, 'Bangalore, India', 23, '7777777777');
```

### Foreign Key Constraint

```sql
ALTER TABLE student_profiles
ADD CONSTRAINT fk_student_id
FOREIGN KEY (student_id)
REFERENCES students(student_id);
```

### Join Query

```sql
SELECT s.student_id, s.name, sp.address, sp.age, sp.phone
FROM students s
JOIN student_profiles sp ON s.student_id = sp.student_id;
```

---

## 📈 One-to-Many Relationship: Students & Marks

### Tables & Data

```sql
CREATE TABLE students (
  student_id SERIAL PRIMARY KEY,
  name VARCHAR(100) NOT NULL
);

CREATE TABLE marks (
  mark_id SERIAL PRIMARY KEY,
  student_id INT,
  subject VARCHAR(50),
  marks INT,
  FOREIGN KEY (student_id) REFERENCES students(student_id)
);

INSERT INTO students (name)
VALUES ('Akarsh Vyas'), ('Simran Mehta'), ('Rohan Gupta');

INSERT INTO marks (student_id, subject, marks)
VALUES
(1, 'English', 85), (1, 'Math', 89), (1, 'Science', 92),
(2, 'English', 80), (2, 'Math', 75), (2, 'Science', 78),
(3, 'English', 72), (3, 'Math', 70), (3, 'Science', 74);
```

---

## 📆 One-to-Many: Products & Orders. (Files are attached in Github)

### Tables

```sql
CREATE TABLE products (
  product_id INT PRIMARY KEY,
  product_name VARCHAR(100),
  category TEXT,
  price NUMERIC(10,2),
  stock_quantity INT,
  is_available BOOLEAN,
  added_on DATE
);

CREATE TABLE orders (
  order_id INT PRIMARY KEY,
  product_id INT,
  quantity INT,
  order_date DATE,
  customer_name VARCHAR(50),
  payment_method VARCHAR(50),
  CONSTRAINT fk_product FOREIGN KEY (product_id)
  REFERENCES products(product_id) ON DELETE CASCADE
);
```

### Sample Queries

#### Q1. Show each order along with the product name and price

```sql
SELECT o.order_id, o.customer_name, p.product_name, p.price
FROM orders o
JOIN products p ON o.product_id = p.product_id;
```

#### Q2. Show all products even if they were never ordered

```sql
SELECT p.product_name, o.order_id
FROM products p
LEFT JOIN orders o ON p.product_id = o.product_id;
```

#### Q3. Show orders for only 'Electronics' category

```sql
SELECT o.order_id, p.product_name, p.category
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE p.category = 'Electronics';
```

#### Q4. List all orders sorted by product price (high to low)

```sql
SELECT o.order_id, p.product_name, p.price
FROM orders o
JOIN products p ON o.product_id = p.product_id
ORDER BY p.price DESC;
```

#### Q5. Show number of orders placed for each product

```sql
SELECT p.product_name, COUNT(o.order_id) AS total_orders
FROM products p
LEFT JOIN orders o ON p.product_id = o.product_id
GROUP BY p.product_name;
```

#### Q6. Show total revenue earned per product

```sql
SELECT p.product_name, SUM(o.quantity * p.price) AS revenue
FROM products p
JOIN orders o ON p.product_id = o.product_id
GROUP BY p.product_name;
```

#### Q7. Show products where total order revenue > ₹2000

```sql
SELECT p.product_name, SUM(o.quantity * p.price) AS total_revenue
FROM products p
JOIN orders o ON p.product_id = o.product_id
GROUP BY p.product_name
HAVING SUM(o.quantity * p.price) > 2000;
```

#### Q8. Show unique customers who ordered 'Fitness' products

```sql
SELECT DISTINCT o.customer_name
FROM orders o
JOIN products p ON o.product_id = p.product_id
WHERE p.category = 'Fitness';
```

---

## 🌐 Many-to-Many: Students & Courses

### Tables & Sample Data

```sql
CREATE TABLE students (
  student_id INT PRIMARY KEY,
  student_name VARCHAR(100)
);

CREATE TABLE courses (
  course_id INT PRIMARY KEY,
  course_name VARCHAR(100)
);

CREATE TABLE student_courses (
  student_id INT,
  course_id INT,
  PRIMARY KEY (student_id, course_id),
  FOREIGN KEY (student_id) REFERENCES students(student_id),
  FOREIGN KEY (course_id) REFERENCES courses(course_id)
);

-- Sample Mapping
INSERT INTO student_courses (student_id, course_id) VALUES
(1, 101), (1, 102), (2, 101), (2, 103), (3, 102);
```

### Queries

#### Q1. Show student and course names

```sql
SELECT s.student_name, c.course_name
FROM student_courses sc
JOIN students s ON sc.student_id = s.student_id
JOIN courses c ON sc.course_id = c.course_id;
```

#### Q2. List all courses taken by 'Simran'

```sql
SELECT c.course_name
FROM student_courses sc
JOIN students s ON sc.student_id = s.student_id
JOIN courses c ON sc.course_id = c.course_id
WHERE s.student_name = 'Simran';
```

---

## 📄 Views

### View 1: Available Fitness Products

```sql
CREATE VIEW available_fitness_products AS
SELECT product_id, name, price, stock_quantity
FROM products
WHERE category = 'Fitness' AND is_available = TRUE;
```

### View 2: Low Stock Products

```sql
CREATE VIEW low_stock_products AS
SELECT name, category, stock_quantity
FROM products
WHERE stock_quantity < 30;
```

---

## ⚖️ Stored Procedure

### Add Product Procedure

```sql
CREATE PROCEDURE add_product(
    p_name VARCHAR,
    p_sku CHAR(8),
    p_price NUMERIC,
    p_qty INT,
    p_category TEXT
)
LANGUAGE plpgsql
AS $$
BEGIN
    INSERT INTO products(name, sku_code, price, stock_quantity, category)
    VALUES (p_name, p_sku, p_price, p_qty, p_category);

    RAISE NOTICE 'Product added successfully!';
END;
$$;
```
