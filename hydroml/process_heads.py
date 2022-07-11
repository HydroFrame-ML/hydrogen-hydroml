import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Optional, Union, List, Mapping

class SaturationHead(nn.Module):
    """
    A module for calculating saturation from pressure
    and associated parameters. Encodes the van Genuchten
    relation:

    $$
    S(p) = \frac{S_{sat} - S_{res}}{\left[ (1 + (a \cdot p)^n)^{1-1/n} \right]} + S_{res}
    $$

    where S(p) is the saturation, p is the pressure,
    a & n are parameter values, and S_{sat} & S_{res}
    are the saturation and residual moisture contents.
    """

    def __init__(self):
        super().__init__()

    def forward(self, pressure, a=None, n=None, s_sat=1.0, s_res=0.0001):
        """
        Forward mode runs prediction directly
        """
        x = (1 - (a * pressure)) ** n
        sgn_x = torch.sign(x)
        scaled_x = torch.abs(x) ** (-1/n)
        saturation = ((s_sat - s_res) * (sgn_x * scaled_x)) + s_res
        saturation = torch.clamp(saturation, 0.0, 1.0)
        saturation = torch.nan_to_num(saturation, 1.0)
        return saturation


class WaterTableDepthHead(nn.Module):
    """
    Calculate water table depth from the land surface
    Adapted as directly as possible from pftools
    """
    def __init__(self):
        super().__init__()

    def forward(self, pressure,saturation,dz):
        """
        :param pressure: A nz-by-ny-by-nx ndarray of pressure values (bottom layer to top layer)
        :param saturation: A nz-by-ny-by-nx ndarray of saturation values (bottom layer to top layer)
        :param dz: An ndarray of shape (nz,) of thickness values (bottom layer to top layer)
        :return: A ny-by-nx ndarray of water table depth values (measured from the top)

        """
           # Handle single-column pressure/saturation values
        if pressure.ndim == 1:
            pressure = pressure[:, None, None]
        if saturation.ndim == 1:
            saturation = saturation[:, None, None]

        domain_thickness = torch.sum(dz)

        # Sentinel values padded to aid in subsequent calculations
        # A layer of thickness 0 at the top
        dz = torch.hstack([dz, torch.tensor(0)])
        # An unsaturated layer at the top
        # pad_width is a tuple of (n_before, n_after) for each dimension
        saturation = F.pad(
            saturation, ((0, 0, 0, 0, 0, 1)), mode='constant', value=0
        )

        # Elevation of the center of each layer from the bottom (bottom layer to top layer)
        _elevation = torch.cumsum(dz,dim=0) - (dz / 2)
        # Make 3D with shape (nz, 1, 1) to allow subsequent operations
        _elevation = _elevation[:, None, None]

        """
        Indices of first unsaturated layer across the grid, going from bottom to top
        with 0 indicating the bottom layer.
        NOTE: np.argmax on a boolean array returns the index of the first True value it encounters.
        It returns a 0 if it doesn't find *any* True values.
        This would normally be a problem - however, since we added a sentinel 0 value to the sat array,
        we will not encounter this case.
        """

        sat_layer = saturation < 1

        z_indices = torch.maximum(
            torch.argmax(sat_layer.float(), axis=0) - 1,  # find first unsaturated layer; back up one cell
            torch.tensor(0)  # clamp min. index value to 0
        )
        # Make 3D with shape (1, ny, nx) to allow subsequent operations
        z_indices = z_indices[None, ...]

        saturation_elevation = torch.take_along_dim(_elevation, z_indices, dim=0)  # shape (1, ny, nx)
        ponding_depth = torch.take_along_dim(pressure, z_indices, dim=0)  # shape (1, ny, nx)

        wt_height = saturation_elevation + ponding_depth  # shape (1, ny, nx)
        wt_height = torch.clip(wt_height, 0, domain_thickness)  # water table height clamped between 0<->domain thickness
        wtd = domain_thickness - wt_height  # shape (1, ny, nx)

        return wtd.squeeze(0)


class SurfaceStorageHead(nn.Module):
    """
    Calculate surface storage on a grid
    Adapted as directly as possible from pftools
    """
    def __init__(self):
        super().__init__()

    def forward(self, pressure, dx, dy, mask=None):
        """
        Calculate gridded surface storage on the top layer.
        Surface storage is given by:
          Pressure at the top layer * dx * dy (for pressure values > 0)
        :param pressure: A nz-by-ny-by-nx ndarray of pressure values (bottom layer to top layer)
        :param dx: Length of a grid element in the x direction
        :param dy: Length of a grid element in the y direction
        :param mask: A nz-by-ny-by-nx ndarray of mask values (bottom layer to top layer)
            If None, assumed to be an nz-by-ny-by-nx ndarray of 1s.
        :return: An ny-by-nx ndarray of surface storage values
        """
        if mask is None:
            mask = torch.ones_like(pressure)

        mask = torch.where(mask > 0, 1, 0)
        surface_mask = mask[-1, ...]
        total = pressure[-1, ...] * dx * dy
        total[total < 0] = 0  # surface storage is 0 when pressure < 0
        total[surface_mask == 0] = 0  # output values for points outside the mask are clamped to 0
        return total


class OverlandFlowHead(nn.Module):
    """
    Calculate Overland Flow
    Adapted as directly as possible from pftools, note that all overland
    flow related functions are methods of this class
    """
    def __init__(self, epsilon=torch.tensor(1e-5)):
        super().__init__()
        self.epsilon = epsilon


    def _overland_flow_kinematic(
        self,
        pressure_top,
        slopex,
        slopey,
        mannings,
        dx,
        dy,
        mask=None,
    ):
        """
        Kinematic
        """
        ny, nx = pressure_top.shape
        # We will be tweaking the slope values so we make a copy
        slopex = torch.clone(slopex)
        slopey = torch.clone(slopey)

        # We're only interested in the surface mask, as an ny-by-nx array
        if mask is None:
            mask = torch.ones_like(pressure_top)
        mask = torch.where(
            mask > torch.tensor(0),
            torch.tensor(1),
            torch.tensor(0)
        )

        # Find all patterns of the form
        #  -------
        # | 0 | 1 |
        #  -------
        # and copy the slopex values from the '1' cells to the corresponding '0' cells
        _x, _y = torch.where(torch.diff(mask, append=torch.zeros((ny,1)), dim=1) == 1)
        slopex[(_x, _y)] = slopex[(_x, _y + 1)]

        # Find all patterns of the form
        #  ---
        # | 0 |
        # | 1 |
        #  ---
        # and copy the slopey values from the '1' cells to the corresponding '0' cells
        _x, _y = torch.where(torch.diff(mask, append=torch.zeros((1,nx)), dim=0) == 1)
        slopey[(_x, _y)] = slopey[(_x + 1, _y)]

        slope = torch.maximum(self.epsilon, torch.hypot(slopex, slopey))

        # Upwind pressure - this is for the north and east face of all cells
        # The slopes are calculated across these boundaries so the upper x/y boundaries are included in these
        # calculations. The lower x/y boundaries are added further down as q_x0/q_y0
        pressure_top_padded = F.pad(pressure_top[:, 1:], (0, 1, 0, 0))  # pad right
        sgn_x = torch.sign(slopex)
        pupwindx = (torch.maximum(torch.tensor(0), sgn_x * pressure_top_padded)
                    + torch.maximum(torch.tensor(0), -sgn_x * pressure_top))

        sgn_y = torch.sign(slopey)
        pressure_top_padded = F.pad(pressure_top[1:, :], (0, 0, 0, 1))  # pad bottom
        pupwindy = (torch.maximum(torch.tensor(0), sgn_y * pressure_top_padded)
                    + torch.maximum(torch.tensor(0), -sgn_y * pressure_top))

        flux_factor = torch.sqrt(slope) * mannings
        # Flux across the x/y directions
        q_x = -slopex / flux_factor * pupwindx ** (5 / 3) * dy
        q_y = -slopey / flux_factor * pupwindy ** (5 / 3) * dx

        # Fix the lower x boundary
        # Use the slopes of the first column
        px = torch.maximum(torch.tensor(0), torch.sign(slopex[:, 0]) * pressure_top[:, 0])
        q_x0 = -slopex[:, 0] / flux_factor[:, 0] *  px ** (5 / 3) * dy
        qeast = torch.hstack([q_x0[:, np.newaxis], q_x])

        # Fix the lower y boundary
        # Use the slopes of the first row
        py = torch.maximum(torch.tensor(0), torch.sign(slopey[0, :]) * pressure_top[0, :])
        q_y0 = -slopey[0, :] / flux_factor[0, :] * py ** (5 / 3) * dx
        qnorth = torch.vstack([q_y0, q_y])

        return qeast, qnorth

    def _overland_flow(self, pressure_top, slopex, slopey, mannings, dx, dy):
        """
        Default implementation of overland flow
        """
        # Calculate fluxes across east and north faces
        zero = torch.tensor(0)

        # ---------------
        # The x direction
        # ---------------
        qx = -((torch.sign(slopex) * (torch.abs(slopex) ** 0.5) / mannings)
               * (pressure_top ** (5 / 3)) * dy)

        # Upwinding to get flux across the east face of cells
        # based on qx[i] if it is positive and qx[i+1] if negative
        qeast = (torch.maximum(zero, qx[:, :-1])
                 - torch.maximum(zero, -qx[:, 1:]))

        # Add the left boundary - pressures outside domain are 0 so flux
        # across this boundary only occurs when qx[0] is negative
        qeast = torch.cat([
            -torch.maximum(zero, -qx[:, slice(0, 1)]),
            qeast,
            torch.maximum(zero, qx[:, slice(-1, None)])
        ], dim=1)

        # ---------------
        # The y direction
        # ---------------
        qy = -((torch.sign(slopey) * (torch.abs(slopey) ** 0.5) / mannings)
               * (pressure_top ** (5 / 3)) * dx)
        # Upwinding to get flux across the north face of cells
        # based in qy[j] if it is positive and qy[j+1] if negative
        qnorth = (torch.maximum(zero, qy[:-1, :])
                  - torch.maximum(zero, -qy[1:, :]))

        # Add the top and bottom boundary - pressures outside domain
        # are 0 so flux across this boundary only occurs when qy[0] is negative
        qnorth = torch.cat([
            -torch.maximum(zero, -qy[slice(0, 1), :]),
            qnorth,
            torch.maximum(zero, qy[slice(-1, None), :])
        ], dim=0)
        return qeast, qnorth


    def calculate_overland_fluxes(
        self,
        pressure,
        slopex,
        slopey,
        mannings,
        dx,
        dy,
        flow_method='OverlandKinematic',
        mask=None
    ):
        """
        Calculate overland fluxes across grid faces
        :param pressure: A nz-by-ny-by-nx ndarray of pressure values (bottom layer to top layer)
        :param slopex: ny-by-nx
        :param slopey: ny-by-nx
        :param mannings: a scalar value, or a ny-by-nx ndarray
        :param dx: Length of a grid element in the x direction
        :param dy: Length of a grid element in the y direction
        :param flow_method: Either 'OverlandFlow' or 'OverlandKinematic'
            'OverlandKinematic' by default.
        :param epsilon: Minimum slope magnitude for solver. Only applicable if flow_method='OverlandKinematic'.
            This is set using the Solver.OverlandKinematic.Epsilon key in Parflow.
        :param mask: A nz-by-ny-by-nx ndarray of mask values (bottom layer to top layer)
            If None, assumed to be an nz-by-ny-by-nx ndarray of 1s.
        :return: A 2-tuple:
            qeast - A ny-by-(nx+1) ndarray of overland flux values
            qnorth - A (ny+1)-by-nx ndarray of overland flux values
        """

        """
        Numpy array origin is at the top left.
        The cardinal direction along axis 0 (rows) is North (going down!!).
        The cardinal direction along axis 1 (columns) is East (going right).
        qnorth (ny+1,nx) and qeast (ny,nx+1) values are to be interpreted as follows.
        +-------------------------------------> (East)
        |
        |                           qnorth_i,j (outflow if negative)
        |                                  +-----+------+
        |                                  |     |      |
        |                                  |     |      |
        |  qeast_i,j (outflow if negative) |-->  v      |---> qeast_i,j+1 (outflow if positive)
        |                                  |            |
        |                                  | Cell  i,j  |
        |                                  +-----+------+
        |                                        |
        |                                        |
        |                                        v
        |                           qnorth_i+1,j (outflow if positive)
        v
        (North)
        """

        pressure_top = torch.clone(pressure[-1, ...])
        pressure_top = torch.nan_to_num(pressure_top)
        pressure_top[pressure_top < 0] = 0

        assert flow_method in ('OverlandFlow', 'OverlandKinematic'), (
                'Unknown flow method')
        if flow_method == 'OverlandKinematic':
            qeast, qnorth = self._overland_flow_kinematic(
                pressure_top, slopex, slopey, mannings, dx, dy, mask
            )
        else:
            qeast, qnorth = self._overland_flow(
                pressure_top, slopex, slopey, mannings, dx, dy
            )

        return qeast, qnorth


    # -----------------------------------------------------------------------------

    def calculate_overland_flow_grid(
        self,
        pressure,
        slopex,
        slopey,
        mannings,
        dx,
        dy,
        flow_method='OverlandKinematic',
        mask=None
    ):
        """
        Calculate overland outflow per grid cell of a domain
        :param pressure: A nz-by-ny-by-nx ndarray of pressure values (bottom layer to top layer)
        :param slopex: ny-by-nx
        :param slopey: ny-by-nx
        :param mannings: a scalar value, or a ny-by-nx ndarray
        :param dx: Length of a grid element in the x direction
        :param dy: Length of a grid element in the y direction
        :param flow_method: Either 'OverlandFlow' or 'OverlandKinematic'
            'OverlandKinematic' by default.
        :param epsilon: Minimum slope magnitude for solver. Only applicable if kinematic=True.
            This is set using the Solver.OverlandKinematic.Epsilon key in Parflow.
        :param mask: A nz-by-ny-by-nx ndarray of mask values (bottom layer to top layer)
            If None, assumed to be an nz-by-ny-by-nx ndarray of 1s.
        :return: A ny-by-nx ndarray of overland flow values
        """

        qeast, qnorth = self.calculate_overland_fluxes(
            pressure,
            slopex,
            slopey,
            mannings,
            dx,
            dy,
            flow_method=flow_method,
            mask=mask
        )

        # Outflow is a positive qeast[i,j+1] or qnorth[i+1,j] or a negative qeast[i,j], qnorth[i,j]
        outflow = (torch.maximum(torch.tensor(0), qeast[:, 1:])
                + torch.maximum(torch.tensor(0), -qeast[:, :-1])
                + torch.maximum(torch.tensor(0), qnorth[1:, :])
                + torch.maximum(torch.tensor(0), -qnorth[:-1, :]))
        if mask is not None:
            # Set the outflow values outside the mask to 0
            if len(mask.shape) == 3:
                mask = mask[-1, ...]
            outflow[mask == 0] = 0
        return outflow


    def calculate_overland_flow(
        self,
        pressure,
        slopex,
        slopey,
        mannings,
        dx,
        dy,
        mask=None,
        flow_method='OverlandKinematic'
    ):
        """
        Calculate overland outflow out of a domain
        :param pressure: A nz-by-ny-by-nx ndarray of pressure values (bottom layer to top layer)
        :param slopex: ny-by-nx
        :param slopey: ny-by-nx
        :param mannings: a scalar value, or a ny-by-nx ndarray
        :param dx: Length of a grid element in the x direction
        :param dy: Length of a grid element in the y direction
        :param flow_method: Either 'OverlandFlow' or 'OverlandKinematic'
            'OverlandKinematic' by default.
        :param epsilon: Minimum slope magnitude for solver. Only applicable if flow_method='OverlandKinematic'.
            This is set using the Solver.OverlandKinematic.Epsilon key in Parflow.
        :param mask: A nz-by-ny-by-nx ndarray of mask values (bottom layer to top layer)
            If None, assumed to be an nz-by-ny-by-nx ndarray of 1s.
        :return: A ny-by-nx ndarray of overland flow values
        """
        nz, nx, ny = pressure.shape
        qeast, qnorth = self.calculate_overland_fluxes(
            pressure,
            slopex,
            slopey,
            mannings,
            dx,
            dy,
            flow_method=flow_method,
            mask=mask
        )

        if mask is not None:
            mask = torch.where(mask > 0, torch.tensor(1), torch.tensor(0))
            if len(mask.shape) == 3:
                surface_mask = mask[-1, ...]  # shape ny, nx
            else:
                surface_mask = mask
        else:
            surface_mask = torch.ones_like(slopex)  # shape ny, nx

        # Important to typecast mask to float to avoid values wrapping around when performing a np.diff
       # surface_mask = surface_mask.astype('float')

        # Find edge pixels for our surface mask along each face - N/S/W/E
        # All of these have shape (ny, nx) and values as 0/1

        # find forward difference of +1 on axis 0
        edge_south = torch.maximum(
            torch.tensor(0),
            torch.diff(surface_mask, dim=0, prepend=torch.zeros((1,nx)))
        )
        # find forward difference of -1 on axis 0
        edge_north = torch.maximum(
            torch.tensor(0),
            -torch.diff(surface_mask, dim=0, append=torch.zeros((1,nx)))
        )
        # find forward difference of +1 on axis 1
        edge_west = torch.maximum(
            torch.tensor(0),
            torch.diff(surface_mask, dim=1, prepend=torch.zeros((ny,1)))
        )
        # find forward difference of -1 on axis 1
        edge_east = torch.maximum(
            torch.tensor(0),
            -torch.diff(surface_mask, dim=1, append=torch.zeros((ny,1)))
        )

        # North flux is the sum of +ve qnorth values (shifted up by one) on north edges
        flux_north = torch.sum(torch.maximum(
            torch.tensor(0),
            torch.roll(qnorth, -1, dims=0)[torch.where(edge_north == 1)]
        ))
        # South flux is the negated sum of -ve qnorth values for south edges
        flux_south = torch.sum(torch.maximum(
            torch.tensor(0),
            -qnorth[torch.where(edge_south == 1)]
        ))
        # West flux is the negated sum of -ve qeast values of west edges
        flux_west = torch.sum(torch.maximum(
            torch.tensor(0),
            -qeast[torch.where(edge_west == 1)]
        ))
        # East flux is the sum of +ve qeast values (shifted left by one) for east edges
        flux_east = torch.sum(torch.maximum(
            torch.tensor(0),
            torch.roll(qeast, -1, dims=1)[torch.where(edge_east == 1)]
        ))

        flux = flux_north + flux_south + flux_west + flux_east
        return flux

    def forward(
        self,
        pressure,
        slopex,
        slopey,
        mannings,
        dx,
        dy,
        flow_method='OverlandKinematic',
        mask=None
    ):
        #default forward is grid, use other methods if you want something else...
        return self.calculate_overland_flow_grid(
            pressure,
            slopex,
            slopey,
            mannings,
            dx,
            dy,
            flow_method=flow_method,
            mask=mask
        )

