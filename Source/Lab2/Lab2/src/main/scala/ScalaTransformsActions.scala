import org.apache.spark.{SparkConf, SparkContext}
/**
  * Created by ganesh on 2/1/17.
  */
object ScalaTransformsActions {
  def main(args: Array[String]): Unit = {

    val sparkConf = new SparkConf().setAppName("SparkTransformsActions").setMaster("local[*]")

    val sc = new SparkContext(sparkConf)
    //Creating RDD's for Freinds list
    val person1_friends = sc.textFile("Person1")
    val person2_friends = sc.textFile("Person2")

    // counting number of freinds for both the persons using "count" action
    val person1_count = person1_friends.count()
    println(s"Total number of friends for person1 are $person1_count")
    val person2_count  = person2_friends.count()
    println(s"Total number of friends for person2 are $person2_count")

    // find mutual friends between two persons using transformation "intersection"
    val mutual = person1_friends.intersection(person2_friends)

    //using "collect" action for returning the dataset as array.
    mutual.collect()
    // printing the mutual freinds using "foreach" action
    mutual.foreach(println)
    println("Are mutual Friends of Person1 and Person2")

    //combining both the friends list using "union" transormation
    val union = person1_friends.union(person2_friends)

    union.distinct().foreach(println)
    //Counting total number unique people in both the list
    val count = union.distinct().count()
    println(s"Total number of unique friends of Person1 and Person2 are $count")





  }


}
