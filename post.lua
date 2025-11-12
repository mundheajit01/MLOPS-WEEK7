-- This script tells wrk how to send a POST request
-- with the correct JSON body for our Iris app.

wrk.method = "POST"
wrk.body   = '{"sepal_length_cm": 5.1, "sepal_width_cm": 3.5, "petal_length_cm": 1.4, "petal_width_cm": 0.2}'
wrk.headers["Content-Type"] = "application/json"