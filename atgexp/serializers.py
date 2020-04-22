from rest_framework import serializers
from .models import Avatar
from drf_extra_fields.fields import Base64ImageField


class ImageSerializers(serializers.ModelSerializer):
	image=Base64ImageField()

	class Meta:
		model = Avatar
		fields = ['image']
	
	def create(self, validated_data):
		print(validated_data)
		image=validated_data.pop('image')
		# data=validated_data.pop('data')
		return Avatar.objects.create(image=image)