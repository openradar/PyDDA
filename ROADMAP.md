# PyDDA Roadmap

In this document we show the details of the PyDDA roadmap. The roadmap shows the next steps for PyDDA and what help is needed for 
contributions so that we can acheive our goal of providing a quality Direct Data Assmilation framework for the meterorological community
to use. Right now, PyDDA currently supports retrievals from an arbitrary number of Doppler radars and can integrate in data from rawinsondes,
HRRR and WRF model runs. We would like improve how PyDDA assimilates data into a retrieval. Furthermore, we would like to make PyDDA
more accessible to the international meteorology community. Our current goals in this regard are:

    * Support for a greater number of high resolution (LES) models such as CM1
    * Support for integrating in data from the Rapid Refresh
    * Coarser resolution reanalyses such as the NCEP reanalysis and ERA Interim would also provide useful information. 
    Due to PyDDA's capability in customizing the weight each dataset has in the retrieval, using a weak constraint against coarse 
    resolution reanalyses would provide a useful background.
    * Support for individual point analyses, such as those from wind profilers and METARs
    * Support for radar data in antenna coordinates
    * Improvements in visualizations
    * Documentation improvements, including better descriptions in the current English version of the documentation 
    and versions of the documentation in non-English languages. 
    
 Do not tell yourself that you can't contribute to PyDDA. This is simply imposter syndrome telling you that you can't contribute. 
 Contributions can be made from people of all skill levels in Python, even none! If there is anyone who would like to make a contribution
 to any of these features, we would be more than welcoming of any additions. All we ask is that you follow the 
 [Code of Conduct](https://github.com/openradar/PyDDA/blob/master/CODE_OF_CONDUCT.md) and that your contributions are in accordance with
 the [Contibutor's Guide](https://openradarscience.org/PyDDA/contributors_guide/index.html), complete with documentation and unit 
 tests where applicable.
