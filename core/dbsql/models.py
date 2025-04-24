# 表定义
from django.db import models

# 用户表sys_user
# class sysUser(models.Model):
#     username = models.CharField(max_length=20, verbose_name="用户名")
#     password = models.CharField(max_length=20, verbose_name="密码")
#     email = models.EmailField(null=False, blank=False, verbose_name="邮箱")
#     # 增加所处城市，便于对应提供合同范本
#     location = models.CharField(max_length=20, verbose_name="所在地区")
#     description = models.TextField(null=True, blank=True, verbose_name="个性标签")
#     # avatar = models.ImageField(null=True, blank=True, upload_to='avater', verbose_name="头像")
#     # 角色：admin-管理员 user-用户 audited-被审核员（可上传相关法律法规或案例）
#     choiceR = (
#         (0, "admin"),
#         (1, "user"),
#         (2, "audited")
#     )
#     role = models.IntegerField(choices=choiceR, default=1, verbose_name="角色标签,0:admin,1:user,2:audited")
#     # 数据新增时间即为创建时间
#     create_time = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")
#
#
#     def to_dict(self):
#         return {
#             'id': self.id,
#             'username': self.username,
#             'email': self.email,
#             'description': self.description,
#             'role': self.role,
#             'create_time': self.create_time.strftime('%Y-%m-%d %H:%M:%S'),
#             'location': self.location,
#         }

# 合同表 sys_contract
class sysContract(models.Model):
    # 合同ID
    Id = models.UUIDField(primary_key=True, verbose_name="合同唯一标识")
    # 合同标题
    Title = models.CharField(max_length=200, verbose_name="合同标题")
    # 合同简介
    Brief = models.TextField(verbose_name="合同简介")
    # 合同标签
    Tags = models.JSONField(null=True, blank=True, verbose_name="合同标签")
    # 发布部门
    Department = models.CharField(max_length=100, verbose_name="发布部门")
    # 是否发布
    IsPublished = models.BooleanField(default=True, verbose_name="是否发布")
    # 修改时间
    ModifiedOn = models.DateTimeField(verbose_name="修改时间")
    # 发布时间
    PublishedOn = models.CharField(max_length=20, verbose_name="发布时间")
    # 是否地方性合同
    IsLocal = models.BooleanField(default=True, verbose_name="是否地方性")
    # 地区（为空表示全国性）
    Region = models.CharField(max_length=50, null=True, blank=True, verbose_name="适用地区")
    # 合同类型
    TYPE_CHOICES = (
        (1, "劳动合同"),
        (2, "租赁合同"),
        (3, "买卖合同"),
        (4, "服务合同"),
        (5, "其他合同")
    )
    Type = models.IntegerField(choices=TYPE_CHOICES, verbose_name="合同类型")
    # 是否最新版本
    IsRecent = models.BooleanField(default=False, verbose_name="是否最新版本")
    # 创建时间
    create_time = models.DateTimeField(auto_now_add=True, verbose_name="创建时间")

    class Meta:
        verbose_name = "合同范本"
        verbose_name_plural = "合同范本"
        db_table = "sys_contract"
        ordering = ["-ModifiedOn"]

    def to_dict(self):
        return {
            'Id': str(self.Id),
            'Title': self.Title,
            'Brief': self.Brief,
            'Tags': self.Tags,
            'Department': self.Department,
            'IsPublished': self.IsPublished,
            'ModifiedOn': self.ModifiedOn.strftime('%Y-%m-%d %H:%M:%S'),
            'PublishedOn': self.PublishedOn,
            'IsLocal': self.IsLocal,
            'Region': self.Region,
            'Type': self.Type,
            'IsRecent': self.IsRecent,
            'create_time': self.create_time.strftime('%Y-%m-%d %H:%M:%S')
        }


printf