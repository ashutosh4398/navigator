from django.db import models

# Create your models here.
class Avatar(models.Model):
	image = models.ImageField(upload_to="images")
	timestamp = models.DateTimeField(auto_now = True,blank = True)
	caption = models.CharField(max_length = 100, blank = True)

	def __str__(self):
		return self.id

	