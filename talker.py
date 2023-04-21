from std_msgs.msg import String
import rospy
import subprocess
import time


def talker():
    time.sleep(0.1)
    pub = rospy.Publisher('chatter', String, queue_size=20)
    rospy.init_node('talker', anonymous=True)
    rate = rospy.Rate(10)
    time.sleep(0.1)

    for i in range(10):

        hello_str = "hello world"
        pub.publish(hello_str)
        rate.sleep()

    
# TODO: publish 10 string data, topic name is chatter

    
    
if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass
