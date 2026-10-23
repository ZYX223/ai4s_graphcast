import unittest
import numpy as np
import xarray as xr
from graphcast import xarray_jax
from data_transfer import to_device
class TransferTests(unittest.TestCase):
 def test_xarray_keeps_dimensions_and_coordinates(self):
  d=xr.Dataset({'x':(('batch','time','lat','lon'),np.ones((1,2,3,4),np.float32))},coords={'time':np.array([-6,0],dtype='timedelta64[h]')})
  got=to_device((d,))[0]
  self.assertEqual(dict(got.sizes),dict(d.sizes))
  np.testing.assert_array_equal(got.time,d.time)
  self.assertTrue(hasattr(xarray_jax.unwrap_data(got.x),'device'))
if __name__=='__main__':unittest.main()
