from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType
from pyspark.sql.window import Window

# Initialize a Spark session
spark_session = SparkSession.builder.appName("Flight Data Analysis Advanced").getOrCreate()

# Load the datasets
flights_data = spark_session.read.csv("flights.csv", header=True, inferSchema=True)
airports_data = spark_session.read.csv("airports.csv", header=True, inferSchema=True)
carriers_data = spark_session.read.csv("carriers.csv", header=True, inferSchema=True)

# Define output file paths
output_directory = "output/"
output_task1 = f"{output_directory}task1_largest_discrepancy.csv"
output_task2 = f"{output_directory}task2_consistent_airlines.csv"
output_task3 = f"{output_directory}task3_canceled_routes.csv"
output_task4 = f"{output_directory}task4_carrier_performance_time_of_day.csv"

# ------------------------
# Task 1: Identify Flights with the Largest Discrepancy in Travel Time
# ------------------------
def find_largest_discrepancy(flights_data, carriers_data):
    # Calculate scheduled and actual travel times in minutes
    flights_data = flights_data.withColumn(
        "scheduled_travel_duration",
        (F.unix_timestamp("ScheduledArrival") - F.unix_timestamp("ScheduledDeparture")) / 60
    ).withColumn(
        "actual_travel_duration",
        (F.unix_timestamp("ActualArrival") - F.unix_timestamp("ActualDeparture")) / 60
    ).withColumn(
        "time_discrepancy",
        F.abs(F.col("scheduled_travel_duration") - F.col("actual_travel_duration"))
    )

    # Join with carriers to retrieve carrier names
    merged_data = flights_data.join(
        carriers_data,
        "CarrierCode"
    ).select(
        "FlightNum", 
        "CarrierName",
        "Origin",
        "Destination",
        "scheduled_travel_duration",
        "actual_travel_duration",
        "time_discrepancy",
        "CarrierCode"
    )

    # Define a window to rank discrepancies for each carrier
    window_specification = Window.partitionBy("CarrierCode").orderBy(F.desc("time_discrepancy"))

    # Apply the window function to get the largest discrepancy per carrier
    largest_discrepancy_data = merged_data.withColumn("rank", F.row_number().over(window_specification)) \
        .filter(F.col("rank") == 1) \
        .select(
            "FlightNum",
            "CarrierName",
            "Origin",
            "Destination",
            "scheduled_travel_duration",
            "actual_travel_duration",
            "time_discrepancy",
            "CarrierCode"
        )

    # Write the result to a CSV file
    largest_discrepancy_data.write.csv(output_task1, header=True)
    print(f"Output for Task 1 written to {output_task1}")

find_largest_discrepancy(flights_data, carriers_data)

# ------------------------
# Task 2: Identify Most Consistently On-Time Airlines Using Standard Deviation
# ------------------------
def find_consistent_airlines(flights_data, carriers_data):
    # Calculate departure delay in minutes
    flights_data = flights_data.withColumn(
        "departure_delay",
        (F.unix_timestamp("ActualDeparture") - F.unix_timestamp("ScheduledDeparture")) / 60
    )
    
    # Aggregate data to get standard deviation and count of flights per carrier
    delay_statistics = flights_data.groupBy("CarrierCode").agg(
        F.count("departure_delay").alias("flight_count"),
        F.stddev("departure_delay").alias("stddev_departure_delay")
    ).filter(F.col("flight_count") > 100)  # Only include carriers with more than 100 flights
    
    # Join with carriers to get carrier names
    delay_statistics = delay_statistics.join(
        carriers_data, "CarrierCode"
    ).select(
        "CarrierName",
        "flight_count",
        "stddev_departure_delay"
    ).orderBy("stddev_departure_delay")  # Order by consistency (smallest standard deviation)

    # Write the result to a CSV file
    delay_statistics.write.csv(output_task2, header=True)
    print(f"Output for Task 2 written to {output_task2}")

# Execute Task 2
find_consistent_airlines(flights_data, carriers_data)

# ------------------------
# Task 3: Identify Routes with the Highest Percentage of Canceled Flights
# ------------------------
def find_canceled_routes(flights_data, airports_data):
    # Identify canceled flights
    flights_data = flights_data.withColumn("is_canceled", F.when(F.col("ActualDeparture").isNull(), 1).otherwise(0))
    

    # Calculate total flights and cancellation rate for each origin-destination pair
    cancellation_statistics = flights_data.groupBy("Origin", "Destination").agg(
        F.count("FlightNum").alias("total_flights"),
        F.sum("is_canceled").alias("canceled_flights")
    ).withColumn(
        "cancellation_rate", 
        (F.col("canceled_flights").cast(DoubleType()) / F.col("total_flights")) * 100
    )

    # Join with airports_data to get names and cities for both origin and destination
    routes_with_airports = cancellation_statistics \
        .join(airports_data.withColumnRenamed("AirportCode", "Origin")
              .withColumnRenamed("AirportName", "OriginAirportName")
              .withColumnRenamed("City", "OriginCity"), "Origin") \
        .join(airports_data.withColumnRenamed("AirportCode", "Destination")
              .withColumnRenamed("AirportName", "DestinationAirportName")
              .withColumnRenamed("City", "DestinationCity"), "Destination") \
        .select(
            "Origin",
            "OriginAirportName",
            "OriginCity",
            "Destination",
            "DestinationAirportName",
            "DestinationCity",
            "cancellation_rate"
        ).orderBy(F.desc("cancellation_rate"))  # Order by highest cancellation rate

    # Write the result to a CSV file
    routes_with_airports.write.csv(output_task3, header=True)
    print(f"Output for Task 3 written to {output_task3}")

# Execute Task 3
find_canceled_routes(flights_data, airports_data)

# ------------------------
# Task 4: Analyze Carrier Performance Based on Time of Day
# ------------------------
def analyze_carrier_performance_time_of_day(flights_data, carriers_data):
    # Extract hour from ScheduledDeparture to categorize time of day
    flights_data = flights_data.withColumn(
        "ScheduledHour", F.hour(F.col("ScheduledDeparture"))
    )

    # Define time of day based on hour
    flights_data = flights_data.withColumn(
        "time_of_day",
        F.when((F.col("ScheduledHour") >= 6) & (F.col("ScheduledHour") < 12), "Morning")
         .when((F.col("ScheduledHour") >= 12) & (F.col("ScheduledHour") < 18), "Afternoon")
         .when((F.col("ScheduledHour") >= 18) & (F.col("ScheduledHour") < 24), "Evening")
         .otherwise("Night")
    )

    # Calculate departure delay in minutes
    flights_data = flights_data.withColumn(
        "departure_delay",
        (F.unix_timestamp("ActualDeparture") - F.unix_timestamp("ScheduledDeparture")) / 60
    )

    # Calculate the average delay by carrier and time of day
    average_delay = flights_data.groupBy("CarrierCode", "time_of_day").agg(
        F.avg("departure_delay").alias("avg_departure_delay")
    )

    # Rank carriers within each time period based on average delay
    window_specification = Window.partitionBy("time_of_day").orderBy("avg_departure_delay")
    ranked_carriers = average_delay.withColumn(
        "rank", F.row_number().over(window_specification)
    )

    # Join with carriers to get the carrier name
    final_result = ranked_carriers.join(
        carriers_data, "CarrierCode"
    ).select(
        "CarrierName", "time_of_day", "avg_departure_delay", "rank"
    ).orderBy("time_of_day", "rank")

    # Write the result to a CSV file
    final_result.write.csv(output_task4, header=True)
    print(f"Output for Task 4 written to {output_task4}")

# Execute Task 4
analyze_carrier_performance_time_of_day(flights_data, carriers_data)

# Stop the Spark session
spark_session.stop()