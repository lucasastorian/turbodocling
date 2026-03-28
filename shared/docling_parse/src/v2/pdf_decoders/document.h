//-*-C++-*-

#ifndef PDF_DOCUMENT_DECODER_H
#define PDF_DOCUMENT_DECODER_H

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <qpdf/QPDF.hh>
//#include <qpdf/QPDFPageObjectHelper.hh>

namespace
{
  using td_parser_profile_clock = std::chrono::steady_clock;

  inline bool td_parser_profile_enabled()
  {
    static bool enabled = []() {
      if(auto const* env = std::getenv("TD_PARSER_PROFILE"))
        {
          return (env[0] != '\0') && !((env[0] == '0') && (env[1] == '\0'));
        }
      return false;
    }();

    return enabled;
  }

  inline long long td_parser_profile_ns(td_parser_profile_clock::duration duration)
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
  }

  inline double td_parser_profile_ms(long long ns)
  {
    return static_cast<double>(ns) / 1000000.0;
  }
}

namespace pdflib
{

  template<>
  class pdf_decoder<DOCUMENT>
  {
  public:

    pdf_decoder();
    pdf_decoder(std::map<std::string, double>& timings_);
    ~pdf_decoder();

    nlohmann::json get();

    int get_number_of_pages() { return number_of_pages; }

    nlohmann::json get_annotations() { return json_annots; }

    nlohmann::json get_meta_xml() { return json_annots["meta_xml"]; }
    nlohmann::json get_table_of_contents() { return json_annots["table_of_contents"]; }
    
    bool process_document_from_file(std::string& _filename);
    bool process_document_from_bytesio(std::string& _buffer);

    void decode_document(std::string page_boundary, bool do_sanitization);

    void decode_document(std::vector<int>& page_numbers, std::string page_boundary, bool do_sanitization);

    nlohmann::json decode_page_original(int page_number,
                                        std::string page_boundary,
                                        bool do_sanitization);

  private:

    void update_qpdf_logger();
    
    void update_timings(std::map<std::string, double>& timings_, bool set_timer);

  private:

    std::string filename;
    std::string buffer; // keep a local copy, in order to not let it expire
    
    std::map<std::string, double> timings;

    QPDF qpdf_document;

    QPDFObjectHandle qpdf_root;
    QPDFObjectHandle qpdf_pages;

    int number_of_pages;    

    //nlohmann::json json_toc; // table-of-contents
    nlohmann::json json_annots;
    nlohmann::json json_document;
  };

  pdf_decoder<DOCUMENT>::pdf_decoder():
    filename(""),
    buffer(""),
    
    timings({}),
    qpdf_document(),
    
    // have compatibulity between QPDF v10 and v11
    qpdf_root(),
    qpdf_pages(),
    
    number_of_pages(-1),

    json_annots(nlohmann::json::value_t::null),
    json_document(nlohmann::json::value_t::null)
  {
    update_qpdf_logger();
  }
  
  pdf_decoder<DOCUMENT>::pdf_decoder(std::map<std::string, double>& timings_):
    filename(""),
    buffer(""),
    
    timings(timings_),
    qpdf_document(),

    // have compatibulity between QPDF v10 and v11
    qpdf_root(),
    qpdf_pages(),
    
    number_of_pages(-1),

    json_annots(nlohmann::json::value_t::null),
    json_document(nlohmann::json::value_t::null)
  {
    update_qpdf_logger();
  }

  pdf_decoder<DOCUMENT>::~pdf_decoder()
  {}

  void pdf_decoder<DOCUMENT>::update_qpdf_logger()
  {
    if(loguru::g_stderr_verbosity==loguru::Verbosity_INFO or
       loguru::g_stderr_verbosity==loguru::Verbosity_WARNING)
      {
	// ignore ...	
      }
    else if(loguru::g_stderr_verbosity==loguru::Verbosity_ERROR or
	    loguru::g_stderr_verbosity==loguru::Verbosity_FATAL)
      {
	qpdf_document.setSuppressWarnings(true);
	//qpdf_document.setMaxWarnings(0); only for later versions ...
      }
    else
      {

      }
  }
  
  nlohmann::json pdf_decoder<DOCUMENT>::get()
  {
    LOG_S(INFO) << "get() [in pdf_decoder<DOCUMENT>]";
    
    {
      json_document["annotations"] = json_annots;
    }
    
    {
      nlohmann::json& timings_ = json_document["timings"];

      for(auto itr=timings.begin(); itr!=timings.end(); itr++)
	{
	  timings_[itr->first] = itr->second;
	}
    }

    return json_document;
  }

  bool pdf_decoder<DOCUMENT>::process_document_from_file(std::string& _filename)
  {
    filename = _filename; // save it    
    LOG_S(INFO) << "start processing '" << filename << "' by qpdf ...";        

    utils::timer timer;
    
    try
      {
        qpdf_document.processFile(filename.c_str());
        LOG_S(INFO) << "filename: " << filename << " processed by qpdf!";        

        qpdf_root  = qpdf_document.getRoot();
        qpdf_pages = qpdf_root.getKey("/Pages");

	json_annots = extract_document_annotations_in_json(qpdf_document, qpdf_root);
	
        number_of_pages = qpdf_pages.getKey("/Count").getIntValue();    
        LOG_S(INFO) << "#-pages: " << number_of_pages;

	nlohmann::json& info = json_document["info"];
	{
	  info["filename"] = filename;
	  info["#-pages"] = number_of_pages;
	}
      }
    catch(const std::exception& exc)
      {
        LOG_S(ERROR) << "filename: " << filename << " can not be processed by qpdf: " << exc.what();        
        return false;
      }

    timings[__FUNCTION__] = timer.get_time();

    return true;
  }
  
  bool pdf_decoder<DOCUMENT>::process_document_from_bytesio(std::string& _buffer)
  {
    buffer = _buffer;    
    LOG_S(INFO) << "start processing buffer of size " << buffer.size() << " by qpdf ...";

    utils::timer timer;
    
    try
      {
	std::string description = "processing buffer";	
        qpdf_document.processMemoryFile(description.c_str(),
					buffer.c_str(), buffer.size());

        LOG_S(INFO) << "buffer processed by qpdf!";        

        qpdf_root  = qpdf_document.getRoot();
        qpdf_pages = qpdf_root.getKey("/Pages");

	json_annots = extract_document_annotations_in_json(qpdf_document, qpdf_root);
	
        number_of_pages = qpdf_pages.getKey("/Count").getIntValue();    
        LOG_S(INFO) << "#-pages: " << number_of_pages;

	nlohmann::json& info = json_document["info"];
	{
	  info["filename"] = filename;
	  info["#-pages"] = number_of_pages;
	}
      }
    catch(const std::exception & exc)
      {
        LOG_S(ERROR) << "filename: " << filename << " can not be processed by qpdf: " << exc.what();        
        return false;
      }

    timings[__FUNCTION__] = timer.get_time();

    return true;
  }
  
  void pdf_decoder<DOCUMENT>::decode_document(std::string page_boundary,
					      bool do_sanitization)
  {
    LOG_S(INFO) << "start decoding all pages ...";        
    utils::timer timer;
    bool const profile_enabled = td_parser_profile_enabled();
    long long total_page_decode_ns = 0;
    long long total_page_get_ns = 0;
    
    nlohmann::json& json_pages = json_document["pages"];
    json_pages = nlohmann::json::array({});
    
    bool set_timer=true;
    
    int page_number=0;
    for(QPDFObjectHandle page : qpdf_document.getAllPages())
      {
	utils::timer page_timer;
	
        pdf_decoder<PAGE> page_decoder(page);

        auto const decode_start =
          profile_enabled ? td_parser_profile_clock::now() : td_parser_profile_clock::time_point{};
        auto timings_ = page_decoder.decode_page(page_boundary, do_sanitization);
        if(profile_enabled)
          {
            total_page_decode_ns +=
              td_parser_profile_ns(td_parser_profile_clock::now() - decode_start);
          }
	update_timings(timings_, set_timer);
	set_timer = false;

        auto const get_start =
          profile_enabled ? td_parser_profile_clock::now() : td_parser_profile_clock::time_point{};
        json_pages.push_back(page_decoder.get());
        if(profile_enabled)
          {
            total_page_get_ns +=
              td_parser_profile_ns(td_parser_profile_clock::now() - get_start);
          }

	std::stringstream ss;
	ss << "decoding page " << page_number++;

	timings[ss.str()] = page_timer.get_time();
      }

    if(profile_enabled)
      {
        std::fprintf(
          stderr,
          "[td-parser] mode=wrapped-all pages=%zu decode_pages=%.3fms page_get=%.3fms total=%.3fms\n",
          qpdf_document.getAllPages().size(),
          td_parser_profile_ms(total_page_decode_ns),
          td_parser_profile_ms(total_page_get_ns),
          1000.0 * timer.get_time());
      }

    timings[__FUNCTION__] = timer.get_time();
  }

  void pdf_decoder<DOCUMENT>::decode_document(std::vector<int>& page_numbers,
					      std::string page_boundary,
					      bool do_sanitization)
  {
    LOG_S(INFO) << "start decoding selected pages ...";        
    utils::timer timer;
    bool const profile_enabled = td_parser_profile_enabled();
    long long total_page_decode_ns = 0;
    long long total_page_get_ns = 0;

    // make sure that we only return the page from the page-numbers
    nlohmann::json& json_pages = json_document["pages"];
    json_pages = nlohmann::json::array({});
      
    std::vector<QPDFObjectHandle> pages = qpdf_document.getAllPages();

    bool set_timer=true; // make sure we override all timings for this page-set
    for(auto page_number:page_numbers)
      {
	utils::timer timer;

	if(0<=page_number and page_number<pages.size())
	  {
	    utils::timer page_timer;
	    
	    pdf_decoder<PAGE> page_decoder(pages.at(page_number));
	    
            auto const decode_start =
              profile_enabled ? td_parser_profile_clock::now() : td_parser_profile_clock::time_point{};
	    auto timings_ = page_decoder.decode_page(page_boundary, do_sanitization);
            if(profile_enabled)
              {
                total_page_decode_ns +=
                  td_parser_profile_ns(td_parser_profile_clock::now() - decode_start);
              }
	    
	    update_timings(timings_, set_timer);
	    set_timer=false;
	    
            auto const get_start =
              profile_enabled ? td_parser_profile_clock::now() : td_parser_profile_clock::time_point{};
	    json_pages.push_back(page_decoder.get());
            if(profile_enabled)
              {
                total_page_get_ns +=
                  td_parser_profile_ns(td_parser_profile_clock::now() - get_start);
              }

	    std::stringstream ss;
	    ss << "decoding page " << page_number;
	    
	    timings[ss.str()] = page_timer.get_time();	    
	  }
	else
	  {
	    LOG_S(WARNING) << "page " << page_number << " is out of bounds ...";        
	    
	    nlohmann::json none;
	    json_pages.push_back(none);
	  }
      }

    if(profile_enabled)
      {
        std::fprintf(
          stderr,
          "[td-parser] mode=wrapped-selected pages=%zu decode_pages=%.3fms page_get=%.3fms total=%.3fms\n",
          page_numbers.size(),
          td_parser_profile_ms(total_page_decode_ns),
          td_parser_profile_ms(total_page_get_ns),
          1000.0 * timer.get_time());
      }

    timings[__FUNCTION__] = timer.get_time();
  }

  nlohmann::json pdf_decoder<DOCUMENT>::decode_page_original(int page_number,
                                                             std::string page_boundary,
                                                             bool do_sanitization)
  {
    utils::timer timer;
    bool const profile_enabled = td_parser_profile_enabled();

    std::vector<QPDFObjectHandle> pages = qpdf_document.getAllPages();
    if(not (0<=page_number and page_number<pages.size()))
      {
        LOG_S(WARNING) << "page " << page_number << " is out of bounds ...";
        return nlohmann::json::value_t::null;
      }

    pdf_decoder<PAGE> page_decoder(pages.at(page_number));

    auto const decode_start =
      profile_enabled ? td_parser_profile_clock::now() : td_parser_profile_clock::time_point{};
    page_decoder.decode_page(page_boundary, do_sanitization);
    auto const decode_ns =
      profile_enabled ? td_parser_profile_ns(td_parser_profile_clock::now() - decode_start) : 0;

    auto const get_start =
      profile_enabled ? td_parser_profile_clock::now() : td_parser_profile_clock::time_point{};
    auto result = page_decoder.get_original();
    auto const get_ns =
      profile_enabled ? td_parser_profile_ns(td_parser_profile_clock::now() - get_start) : 0;

    if(profile_enabled)
      {
        std::fprintf(
          stderr,
          "[td-parser] mode=original-only page=%d decode_page=%.3fms page_get_original=%.3fms total=%.3fms\n",
          page_number,
          td_parser_profile_ms(decode_ns),
          td_parser_profile_ms(get_ns),
          1000.0 * timer.get_time());
      }

    return result;
  }

  void pdf_decoder<DOCUMENT>::update_timings(std::map<std::string, double>& timings_,
					     bool set_timer)
  {
    for(auto itr=timings_.begin(); itr!=timings_.end(); itr++)
      {
	if(timings.count(itr->first)==0 or set_timer)
	  {
	    timings[itr->first] = itr->second;
	  }
	else
	  {
	    timings[itr->first] += itr->second;
	  }
      }    
  }

}

#endif
