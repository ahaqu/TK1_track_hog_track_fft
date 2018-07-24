#ifndef STRUCT_H
#define STRUCT_H

#include <iostream>
#include <algorithm>

template <typename T>
class Rect
{
public:
	Rect() :
	  m_xMin(0),
		  m_yMin(0),
		  m_width(0),
		  m_height(0)
	  {
	  }

	  Rect(T xMin, T yMin, T width, T height) :
	  m_xMin(xMin),
		  m_yMin(yMin),
		  m_width(width),
		  m_height(height)
	  {
	  }

	  template <typename T2>
	  Rect(const Rect<T2>& rOther) :
	  m_xMin((T)rOther.XMin()),
		  m_yMin((T)rOther.YMin()),
		  m_width((T)rOther.Width()),
		  m_height((T)rOther.Height())
	  {
	  }

	  inline void Set(T xMin, T yMin, T width, T height)
	  {
		  m_xMin = xMin;
		  m_yMin = yMin;
		  m_width = width;
		  m_height = height;
	  }

	  inline T XMin() const { return m_xMin; }
	  inline void SetXMin(T val) { m_xMin = val; }
	  inline T YMin() const { return m_yMin; }
	  inline void SetYMin(T val) { m_yMin = val; }
	  inline T Width() const { return m_width; }
	  inline void SetWidth(T val) { m_width = val; }
	  inline T Height() const { return m_height; }
	  inline void SetHeight(T val) { m_height = val; }

	  inline void Translate(T x, T y) { m_xMin += x; m_yMin += y; }

	  inline T XMax() const { return m_xMin + m_width; }
	  inline T YMax() const { return m_yMin + m_height; }
	  inline float XCentre() const { return (float)m_xMin + (float)m_width / 2; }
	  inline float YCentre() const { return (float)m_yMin + (float)m_height / 2; }
	  inline T Area() const { return m_width * m_height; }

	  template <typename T2>
	  friend std::ostream& operator <<(std::ostream &rOS, const Rect<T2>& rRect);

	  template <typename T2>
	  float Overlap(const Rect<T2>& rOther) const;

	  template <typename T2>
	  bool IsInside(const Rect<T2>& rOther) const;

public:
	T m_xMin;
	T m_yMin;
	T m_width;
	T m_height;
};

typedef Rect<int> IntRect;
typedef Rect<float> FloatRect;



#endif