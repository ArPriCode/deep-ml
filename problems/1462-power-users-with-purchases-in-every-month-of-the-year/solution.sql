SELECT user_id
FROM purchases
WHERE purchase_date >= '2024-01-01'
  AND purchase_date < '2025-01-01'
GROUP BY user_id
HAVING COUNT(*) >= 24
   AND COUNT(DISTINCT EXTRACT(MONTH FROM purchase_date)) = 12
ORDER BY user_id;