---
layout: post
title: Mybatis分页插件PageHelper的使用
date: 2017-08-30 11:00:00.000000000 +09:00
tags: Mybatis PageHelper
---

[TOC]

## 1.综述

SQL分页查询时大家想到的可能有limit子句，但是在SSM框架中，PageHelper插件为Mybatis分页查询提供了很便捷的接口。PageHelper使用ThreadLocal获取线程中的变量信息，多线程下，不同Thread的ThreadLocal相互隔离，PageHelper通过拦截器获取到同一线程中预编译好的SQL语句，再把SQL语句包装成具有分页功能的SQL语句，并将其再次赋值给下一步操作。但需要注意的是，**PageHelper只对紧跟着的第一个SQL语句起作用**。

## 2.配置

### 使用环境

IDE：IntelliJ IDEA2016

依赖：Maven3

Mybatis：1.3.2

### 配置步骤

* pom.xml中添加PageHelper插件依赖

```xml
<dependency>
	<groupId>com.github.pagehelper</groupId>
	<artifactId>pagehelper</artifactId>
	<version>4.2.1</version>
</dependency>
```



> 注意尽量使用较新的版本，否则版本较低时可能涉及某些接口的方法已经做了较大的修改，会遇到NoSuchMethod之类的错误。

* 在Mybatis的配置文件中配置PageHelper插件

```xml
 <!-- 配置分页插件 -->
    <plugins>
        <plugin interceptor="com.github.pagehelper.PageHelper">
            <property name="dialect" value="mysql" />
            <!-- 该参数默认为false -->
            <!-- 设置为true时，会将RowBounds第一个参数offset当成pageNum页码使用 -->
            <!-- 和startPage中的pageNum效果一样 -->
            <property name="offsetAsPageNum" value="true" />
            <!-- 该参数默认为false -->
            <!-- 设置为true时，使用RowBounds分页会进行count查询 -->
            <property name="rowBoundsWithCount" value="true" />
            <!-- 设置为true时，如果pageSize=0或者RowBounds.limit = 0就会查询出全部的结果 -->
            <!-- （相当于没有执行分页查询，但是返回结果仍然是Page类型） -->
            <property name="pageSizeZero" value="true" />
            <!-- 3.3.0版本可用 - 分页参数合理化，默认false禁用 -->
            <!-- 启用合理化时，如果pageNum<1会查询第一页，如果pageNum>pages会查询最后一页 -->
            <!-- 禁用合理化时，如果pageNum<1或pageNum>pages会返回空数据 -->
            <property name="reasonable" value="false" />
            <!-- 3.5.0版本可用 - 为了支持startPage(Object params)方法 -->
            <!-- 增加了一个`params`参数来配置参数映射，用于从Map或ServletRequest中取值 -->
            <!-- 可以配置pageNum,pageSize,count,pageSizeZero,reasonable,不配置映射的用默认值 -->
            <!-- 不理解该含义的前提下，不要随便复制该配置 -->
            <property name="params" value="pageNum=start;pageSize=limit;" />
        </plugin>
    </plugins>
```

## 3.使用

* 在对应的Mapper.xml中写好SQL语句，不需要使用Limit等子句

```xml
<select id="selectAllJobs" resultType="map">
        SELECT id jobId, job_name jobName, build_count buildCount, build_state state, created_at createdAt FROM job
        WHERE user_id =#{userId}
        ORDER BY job.id DESC
</select>
```

> 此处使用order by子句可以在分页时按升/降序查询。

* 在对应的Controller中调用分页接口方法

```java
//page:查询的页码，即第几页；pageSize：每页返回数据量，即多少条
//此处page来自前端发出的请求，pageSize＝10
List<Job> jobList = null;
PageHelper.startPage(page, pageSize);//PageHelper分页,默认每页10条数据
jobList = jobService.listAllJobs(userId);
//pageInfo获取分页信息
PageInfo<Job> pageInfo = new PageInfo<>(jobList);
int totalPage = pageInfo.getPages();     //总页数
//包装成JSON待返回给前端
JSONObject data = new JSONObject();
data.put("jobs", jobList);
data.put("curPage", page);
data.put("totalPage", totalPage);
```

至此PageHelper的基本使用流程介绍完毕，更多的接口方法大家可以去查看PageHelper的API。



