#ifndef PDF_H
#define PDF_H

#include "onb.h"
#include "hittable.h"
#include "hittableList.h"
#include "utility2.h"
#include "vec3_0.h"
#include <cmath>
#include "pdf.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>



class pdf {
public:
    __device__ __host__ ~pdf() {}

    __device__ virtual float value(const vec3& direction) const = 0;
    __device__ virtual vec3 generate() const = 0;
};

class sphere_pdf : public pdf {
public:
    __device__ __host__ sphere_pdf() {}

    __device__ float value(const vec3& direction) const override {
        return 1 / (4 * pi);
    }

    __device__ vec3 generate() const override {
        return random_unit_vector();
    }
};

class cosine_pdf : public pdf {
public:
    __device__ __host__ cosine_pdf(const vec3& w) : uvw(w) {}

    __device__ float value(const vec3& direction) const override {
        auto cosine_theta = dot(unit_vector(direction), uvw.w());
        return fmax(0.0f, cosine_theta / pi);
    }

    __device__ vec3 generate() const override {
        return uvw.transform(random_cosine_direction());
    }

private:
    onb uvw;
};

class hittable_pdf : public pdf {
public:
    __device__ __host__ hittable_pdf(const hittable& objects, const point3& origin)
        : objects(&objects), origin(origin)
    {
    }

    __device__ float value(const vec3& direction) const override {
        float result = objects->pdf_value(origin, direction);
        return result;
    }

    __device__ vec3 generate() const override {
        return objects->random(origin);
    }

    __device__ void set_origin(const point3& o) {
        origin = o;
    }
private:
    const hittable* objects;
    point3 origin;
};


class mixture_pdf : public pdf {
public:
    __device__ __host__ mixture_pdf(pdf& p0, pdf& p1)
        : pdf0(p0), pdf1(p1)
    {
    }


    __device__ float value(const vec3& direction) const override {
        return 0.5 * pdf0.value(direction) + 0.5 * pdf1.value(direction);
    }

    __device__ vec3 generate() const override {
        if (random_double() < 0.5)
            return pdf0.generate();
        else
            return pdf1.generate();
    }

    __device__ void set_pdfs(pdf& p0, pdf& p1) {
        pdf0 = p0;
        pdf1 = p1;
    }

private:
    pdf& pdf0;
    pdf& pdf1;
};

struct pdf_record {
    __device__ pdf_record() : type(NONE) {}
    __device__ ~pdf_record() {
        // Manually call the destructor for the active union member
        switch (type) {
        case SPHERE:
            sphere.~sphere_pdf();
            break;
        case COSINE:
            cosine.~cosine_pdf();
            break;
        case HITTABLE:
            the_hittable_pdf.~hittable_pdf();
            break;
            //case MIXTURE:
            //    mixture.~mixture_pdf();
            //    break;
        default:
            break;
        }
    }

    enum pdf_type { NONE, SPHERE, COSINE, HITTABLE, MIXTURE } type;
    union {
        sphere_pdf sphere;
        cosine_pdf cosine;
        hittable_pdf the_hittable_pdf;
        //mixture_pdf mixture;
    };

    __device__ void init_sphere() {
        type = SPHERE;
        new(&sphere) sphere_pdf();
    }

    __device__ void init_cosine(const vec3& n) {
        type = COSINE;
        new(&cosine) cosine_pdf(n);
    }

    __device__ void init_hittable(const hittable& the_hittable, const point3& origin) {
        type = HITTABLE;
        new(&the_hittable_pdf) hittable_pdf(the_hittable, origin);
    }

    __device__ void init_none() {
        type = NONE;
    }

    /*__device__ void init_mixture(pdf& p0, pdf& p1) {
        type = MIXTURE;
        new(&mixture) mixture_pdf(p0, p1);
    }*/

    __device__ float value(const vec3& d) const {
        switch (type) {
        case NONE: return 1.0f;
        case SPHERE: return sphere.value(d);
        case COSINE: return cosine.value(d);
        case HITTABLE: return the_hittable_pdf.value(d);
            //case MIXTURE: return mixture.value(d);
        default: return 1.0f;
        }
    }

    __device__ vec3 generate() const {
        switch (type) {
        case NONE: return random_unit_vector();
        case SPHERE: return sphere.generate();
        case COSINE: return cosine.generate();
        case HITTABLE: return the_hittable_pdf.generate();
            //case MIXTURE: return mixture.generate();
        default: return vec3(0, 0, 0);
        }
    }

    __device__ pdf& get_pdf() {
        switch (type) {
        case NONE: return sphere; // Return a default pdf to avoid compiler warnings
        case SPHERE: return sphere;
        case COSINE: return cosine;
        case HITTABLE: return the_hittable_pdf;
            //case MIXTURE: return mixture;
        default: return sphere; // Return a default pdf to avoid compiler warnings
        }
    }
};


#endif