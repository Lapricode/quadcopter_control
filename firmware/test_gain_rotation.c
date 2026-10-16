// Standalone check of lqr_rotate_gain_with_yaw (copied verbatim from the firmware patch).
// Reads K (48 floats) and yaw from argv/stdin, prints the rotated gain.
#include <stdio.h>
#include <math.h>
static void lqr_rotate_gain_with_yaw(const float K[4][12], float yaw, float Kout[4][12])
{
  const float c = cosf(yaw), s = sinf(yaw);
  for (int i = 0; i < 4; ++i)
  {
    for (int j = 0; j < 12; ++j)
      Kout[i][j] = K[i][j];
    Kout[i][0] = c * K[i][0] - s * K[i][1];
    Kout[i][1] = s * K[i][0] + c * K[i][1];
  }
}
int main(void)
{
  float K[4][12], Ko[4][12], yaw;
  if (scanf("%f", &yaw) != 1) return 1;
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 12; ++j) if (scanf("%f", &K[i][j]) != 1) return 1;
  lqr_rotate_gain_with_yaw(K, yaw, Ko);
  for (int i = 0; i < 4; ++i) for (int j = 0; j < 12; ++j) printf("%.9g\n", Ko[i][j]);
  return 0;
}
