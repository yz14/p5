---
layout: post
title:  "模板"
date:   2024-04-30 16:35:00 +0530
categories: jekyll update
image: /images/image-1.png
categories: [one, two]
---
使用的一些规范：  
我将所有的代码全部都放在：`_posts`文件夹下。为了方便阅读，我还把代码写成`.md`的格式，并在里面写了一些理解，也都放在`_posts`文件夹下。在将`.py`文件写成`.md`文档时，我使用了下面的方法来展示代码：  
{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}  
如果需要返回首页，请点击[这里][link1].

[link1]: https://yz14.github.io/p5
