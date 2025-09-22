#ifndef MATERIAL_H
#define MATERIAL_H

#include <string> 
#include "color.h"
#include <stdexcept>
#include "texture.h"
#include "onb.h"
#include "utility2.h"
#include "vec3_0.h"
#include "ray.h" 
#include "hittable.h"
#include "pdf.h"
#include <cmath>


class __align__(16) Material
{
public:
	__device__ __host__ Material() {
	}

	__device__ __host__ Material(bool isEmissive = false, texture * tex = new solid_color(color(0, 0, 0)), float probability_illumination = 1.0f, float brightness = 4.0f, float specularRate = 0.0f, float roughness = 0.0f, float refractRate = 0.0f, float indexOfRefraction = 1.0f, float refract_roughness = 0.0f)
		: isEmissive(isEmissive), matTexture(tex), probability_illumination(probability_illumination), brightness(brightness), specularRate(specularRate), roughness(roughness), refractRate(refractRate), indexOfRefraction(indexOfRefraction), refract_roughness(refract_roughness) {
	}

	__device__ __host__ Material(bool isEmissive = false, color materialColor = color(0, 0, 0), float probability_illumination = 1.0f, float brightness = 4.0f, float specularRate = 0.0f, float roughness = 0.0f, float refractRate = 0.0f, float indexOfRefraction = 1.0f, float refract_roughness = 0.0f)
		: Material(isEmissive, new solid_color(materialColor), probability_illumination, brightness, specularRate, roughness, refractRate, indexOfRefraction, refract_roughness) {
	}

	__device__ __host__ Material(const Material&) = default;
	__device__ __host__ Material& operator=(const Material&) = default;

	bool isEmissive;
	color materialColor;
	float specularRate;
	float roughness;
	float refractRate;
	float indexOfRefraction;
	float refract_roughness;
	float brightness;
	float probability_illumination;




	__device__ bool scatter(const ray & r_in, const hit_record & rec, color & attenuation, ray & scattered, pdf_record & p_rec)
	{


		if (isEmissive && random_double() < probability_illumination) 
		{
			if (dot(rec.normal, r_in.direction()) < 0)
				attenuation = brightness * matTexture->value(rec.u, rec.v, rec);
			else
				attenuation = color(0, 0, 0);
			return false;
		}

		float r = random_double();
		if (r < specularRate)
		{
			specularReflection(r_in, rec, attenuation, scattered, p_rec);
		}
		else if (specularRate <= r && r <= specularRate + refractRate)
		{
			refraction(r_in, rec, attenuation, scattered, p_rec);
		}
		else
		{
			lambertianReflection(r_in, rec, attenuation, scattered, p_rec);
		}


		return true;
	}

	__device__ color emitted(const hit_record & rec, const ray & r_in)
	{
		if (isEmissive) {
			if (dot(rec.normal, r_in.direction()) < 0)
				return matTexture->value(rec.u, rec.v, rec);
			else
				return color(0, 0, 0);
		}
		else {
			return color(0, 0, 0);
		}
	}


	__device__ void lambertianReflection(const ray & r_in, const hit_record & rec, color & attenuation, ray & scattered, pdf_record & p_rec)
	{
		onb basis(rec.normal);

		auto scatter_direction = basis.transform(random_cosine_direction());


		// Catch degenerate scatter direction
		if (scatter_direction.near_zero())
			scatter_direction = rec.normal;

		scattered = ray(rec.p, scatter_direction);

		attenuation = matTexture->value(rec.u, rec.v, rec);
		set_scattering_pdf(r_in, rec, scattered, p_rec, 0);
	}

	__device__ void specularReflection(const ray & r_in, const hit_record & rec, color & attenuation, ray & scattered, pdf_record & p_rec)
	{
		vec3 reflected = reflect(r_in.direction(), rec.normal);
		vec3 roughened_reflected = lerp(reflected, random_unit_vector(), roughness);

		if (roughened_reflected.near_zero())
			roughened_reflected = reflected;

		scattered = ray(rec.p, roughened_reflected);
		attenuation = matTexture->value(rec.u, rec.v, rec);
		set_scattering_pdf(r_in, rec, scattered, p_rec, 1);
	}

	__device__ void refraction(const ray & r_in, const hit_record & rec, color & attenuation, ray & scattered, pdf_record & p_rec)
	{
		vec3 unit_direction = unit_vector(r_in.direction());
		bool front_face = dot(unit_direction, rec.normal) < 0;
		vec3 n = front_face ? rec.normal : -rec.normal;
		float ri = front_face ? (1.0 / indexOfRefraction) : indexOfRefraction;

		float cos_theta = fminf(dot(-unit_direction, n), 1.0);
		float sin_theta = sqrtf(1.0 - cos_theta * cos_theta);
		bool cannot_refract = sin_theta * ri > 1.0;
		cannot_refract = cannot_refract || reflectance(cos_theta, ri) > random_double();

		vec3 result;
		if (!cannot_refract)
		{
			result = refract(unit_direction, n, ri);
		}
		else
		{
			result = reflect(unit_direction, n);
		}
		attenuation = matTexture->value(rec.u, rec.v, rec);
		scattered = ray(rec.p, unit_vector(result));
		set_scattering_pdf(r_in, rec, scattered, p_rec, 2);
	}

	__device__ static float reflectance(float cosine, float refraction_index) {
		// Use Schlick's approximation for reflectance.
		float r0 = (1 - refraction_index) / (1 + refraction_index);
		r0 = r0 * r0;
		return r0 + (1 - r0) * powf((1 - cosine), 5);
	}


	__device__ void set_scattering_pdf(const ray & r_in, const hit_record & rec,
		const ray & scattered, pdf_record & p_rec, int type)
	{
		switch (type)
		{
		case 0: // cosine-weighted diffuse
		{
			//auto cos_theta = dot(rec.normal, unit_vector(scattered.direction()));
			p_rec.init_cosine(rec.normal);
		}
		break;

		case 1: // perfect reflection
		{
			p_rec.init_none();
		}
		break;

		case 2: // perfect refraction
		{
			p_rec.init_none();
		}
		break;

		default:
			printf("Error: Invalid scattering type in set_scattering_pdf\n");
			break;
		}
	}
private:
	texture* matTexture; // Optional texture for the material
};

namespace Materials
{
	//noise_texture NOISE4(4);
	//noise_texture NOISE8(8);

	//checker_texture CHECKER_GRAY_WHITE(40, color(.5, .5, .5), color(.9, .9, .9));



	/*solid_color WHITE_TEXTURE(Colors::WHITE);
	solid_color RED_TEXTURE(Colors::RED);
	solid_color GREEN_TEXTURE(Colors::GREEN);
	solid_color BLUE_TEXTURE(Colors::BLUE);
	solid_color BLACK_TEXTURE(Colors::BLACK);
	solid_color GREY_TEXTURE(Colors::GREY);
	solid_color SKY_BLUE_TEXTURE(color(0.486, 0.722, 1.0));
	solid_color GRASS_GREEN_TEXTURE(Colors::GRASS_GREEN);*/

	/*Material WHITE_GLASS("white_glass", false, make_shared<solid_color>(WHITE_TEXTURE), 0.0f, 0.0f, 1.0f, 1.5f, 0.0f);
	Material RED_GLASS("red_glass", false, make_shared<solid_color>(RED_TEXTURE), 0.0f, 0.0f, 1.0f, 1.5f, 0.0f);
	Material GREEN_GLASS("green_glass", false, make_shared<solid_color>(GREEN_TEXTURE), 0.0f, 0.0f, 1.0f, 1.5f, 0.0f);
	Material BLUE_GLASS("blue_glass", false, make_shared<solid_color>(BLUE_TEXTURE), 0.0f, 0.0f, 1.0f, 1.5f, 0.0f);
	Material WHITE_MATTE("white_matte", false, make_shared<solid_color>(WHITE_TEXTURE));
	Material RED_MATTE("red_matte", false, make_shared<solid_color>(RED_TEXTURE));
	Material GREEN_MATTE("green_matte", false, make_shared<solid_color>(GREEN_TEXTURE));
	Material BLUE_MATTE("blue_matte", false, make_shared<solid_color>(BLUE_TEXTURE));
	Material BLACK_MATTE("black_matte", false, make_shared<solid_color>(BLACK_TEXTURE));
	Material GREY_MATTE("gray_matte", false, make_shared<solid_color>(GREY_TEXTURE));
	Material MIRROR("mirror", false, make_shared<solid_color>(WHITE_TEXTURE), 1.0f, 0.0f);
	Material WHITE_EMISSIVE("white_emissive", true, make_shared<solid_color>(WHITE_TEXTURE), 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	Material RED_EMISSIVE("red_emissive", true, make_shared<solid_color>(RED_TEXTURE), 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	Material GREEN_EMISSIVE("green_emissive", true, make_shared<solid_color>(GREEN_TEXTURE), 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	Material BLUE_EMISSIVE("blue_emissive", true, make_shared<solid_color>(BLUE_TEXTURE), 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	Material SKY_BLUE_EMISSIVE("sky_blue_emissive", true, make_shared<solid_color>(SKY_BLUE_TEXTURE), 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
	Material FRESH_LAWN("fresh_lawn", false, make_shared<solid_color>(GRASS_GREEN_TEXTURE));
	Material WHITE_GLOSSY("white_glossy", false, make_shared<solid_color>(WHITE_TEXTURE), 0.5f, 0.3f, 0.0f, 0.0f, 0.0f);
	Material BASE_PLANE("base_plane", false, make_shared<checker_texture>(CHECKER_GRAY_WHITE), 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
	Material BASE_PLANE_GLOSSY("base_plane_glossy", false, make_shared<checker_texture>(CHECKER_GRAY_WHITE), 0.8f, 0.2f, 0.0f, 0.0f, 0.0f);*/

}


#endif