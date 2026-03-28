//-*-C++-*-

#ifndef PDF_STREAM_DECODER_H
#define PDF_STREAM_DECODER_H

#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace pdflib
{

  using td_stream_profile_clock = std::chrono::steady_clock;

  inline bool td_stream_profile_enabled()
  {
    static bool enabled = []() {
      if(auto const* env = std::getenv("TD_STREAM_PROFILE"))
        {
          return (env[0] != '\0') && !((env[0] == '0') && (env[1] == '\0'));
        }
      return false;
    }();

    return enabled;
  }

  inline bool td_stream_force_copy_resources()
  {
    static bool enabled = []() {
      if(auto const* env = std::getenv("TD_STREAM_FORCE_COPY_RESOURCES"))
        {
          return (env[0] != '\0') && !((env[0] == '0') && (env[1] == '\0'));
        }
      return false;
    }();

    return enabled;
  }

  inline long long td_stream_profile_ns(td_stream_profile_clock::duration duration)
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
  }

  inline double td_stream_profile_ms(long long ns)
  {
    return static_cast<double>(ns) / 1000000.0;
  }

  struct td_stream_profile_data
  {
    size_t top_level_streams{0};
    size_t operator_count{0};
    size_t path_skip_count{0};
    size_t do_count{0};
    size_t do_form_count{0};
    size_t do_image_count{0};
    long long top_level_decode_ns{0};
    long long top_level_interpret_ns{0};
    long long do_total_ns{0};
    long long do_copy_resources_ns{0};
    long long do_parse_stream_ns{0};
    long long do_update_stack_ns{0};
    long long do_nested_interpret_ns{0};
  };

  inline td_stream_profile_data& td_stream_profile()
  {
    static thread_local td_stream_profile_data profile;
    return profile;
  }

  inline void td_stream_profile_reset()
  {
    td_stream_profile() = td_stream_profile_data{};
  }

  inline void td_stream_profile_log(char const* label)
  {
    if(not td_stream_profile_enabled())
      {
        return;
      }

    auto const& profile = td_stream_profile();
    std::fprintf(
      stderr,
      "[td-stream] label=%s top_level_streams=%zu operators=%zu path_skips=%zu "
      "do=%zu do_forms=%zu do_images=%zu top_decode=%.3fms top_interpret=%.3fms "
      "do_total=%.3fms do_copy_resources=%.3fms do_parse_stream=%.3fms "
      "do_update_stack=%.3fms do_nested_interpret=%.3fms\n",
      label,
      profile.top_level_streams,
      profile.operator_count,
      profile.path_skip_count,
      profile.do_count,
      profile.do_form_count,
      profile.do_image_count,
      td_stream_profile_ms(profile.top_level_decode_ns),
      td_stream_profile_ms(profile.top_level_interpret_ns),
      td_stream_profile_ms(profile.do_total_ns),
      td_stream_profile_ms(profile.do_copy_resources_ns),
      td_stream_profile_ms(profile.do_parse_stream_ns),
      td_stream_profile_ms(profile.do_update_stack_ns),
      td_stream_profile_ms(profile.do_nested_interpret_ns));
  }

  template<>
  class pdf_decoder<STREAM>
  {

  public:

    pdf_decoder(pdf_resource<PAGE_DIMENSION>& page_dimension_,
                pdf_resource<PAGE_CELLS>&     page_cells_,
                pdf_resource<PAGE_LINES>&     page_lines_,                                   
                pdf_resource<PAGE_IMAGES>&    page_images_,

                pdf_resource<PAGE_FONTS>&     page_fonts_,                                   
                pdf_resource<PAGE_GRPHS>&     page_grphs_,                                   
                pdf_resource<PAGE_XOBJECTS>&  page_xobjects_);

    ~pdf_decoder();

    void print();

    std::set<std::string> get_unknown_operators();

    // decode the qpdf-stream
    void decode(QPDFObjectHandle& content);

    // methods used to interprete the stream
    void interprete(std::vector<qpdf_instruction>& parameters);

  private:

    bool update_stack(std::vector<pdf_state<GLOBAL> >& stack_,
                      int                              stack_count_);

    void interprete(std::vector<qpdf_instruction>& stream_,
                    std::vector<qpdf_instruction>& parameters_);

    void interprete_stream(std::vector<qpdf_instruction>& parameters);

    pdf_state<GLOBAL>& cgs(); // get current global state
    pdf_state<TEXT>&   cts(); // get current text state
    pdf_state<LINE>&   cls(); // get current line state
    pdf_state<GRPH>&   cgrs(); // get current graphics state

    void q();
    void Q();

    void execute_operator(const qpdf_instruction& op, std::vector<qpdf_instruction>& parameters);
    
  private:

    pdf_resource<PAGE_DIMENSION>& page_dimension;
    pdf_resource<PAGE_CELLS>&     page_cells;
    pdf_resource<PAGE_LINES>&     page_lines;
    pdf_resource<PAGE_IMAGES>&    page_images;

    pdf_resource<PAGE_FONTS>&     page_fonts;
    pdf_resource<PAGE_GRPHS>&     page_grphs;
    pdf_resource<PAGE_XOBJECTS>&  page_xobjects;

    std::set<std::string> unknown_operators;

    std::vector<qpdf_instruction> stream;
    std::vector<pdf_state<GLOBAL> > stack;

    int stack_count;
  };

  pdf_decoder<STREAM>::pdf_decoder(pdf_resource<PAGE_DIMENSION>& page_dimension_,
                                   pdf_resource<PAGE_CELLS>&     page_cells_,
                                   pdf_resource<PAGE_LINES>&     page_lines_,                                   
                                   pdf_resource<PAGE_IMAGES>&    page_images_,
                                   
                                   pdf_resource<PAGE_FONTS>&     page_fonts_,
                                   pdf_resource<PAGE_GRPHS>&     page_grphs_,

                                   pdf_resource<PAGE_XOBJECTS>&  page_xobjects_):
    page_dimension(page_dimension_),
    page_cells(page_cells_),    
    page_lines(page_lines_),
    page_images(page_images_),

    page_fonts(page_fonts_),
    page_grphs(page_grphs_),

    page_xobjects(page_xobjects_),

    unknown_operators({}),
    stream({}),
    stack({}),
    
    stack_count(0)
  {
    LOG_S(INFO) << __FUNCTION__;
  }

  pdf_decoder<STREAM>::~pdf_decoder()
  {
    if(unknown_operators.size()>0)
      {
	LOG_S(WARNING) << "============= ~pdf_decoder ===================";
	for(auto item:unknown_operators)
	  {
	    LOG_S(WARNING) << "unknown operator: " << item;
	  }
	LOG_S(WARNING) << "==============================================";
      }
  }

  std::set<std::string> pdf_decoder<STREAM>::get_unknown_operators()
  {
    LOG_S(INFO) << __FUNCTION__;
    return unknown_operators;
  }

  void pdf_decoder<STREAM>::print()
  {
    LOG_S(INFO) << __FUNCTION__;
    for(auto row:stream)
      {
        LOG_S(INFO) << std::setw(12) << row.key << " | " << row.val;
      }
  }

  void pdf_decoder<STREAM>::decode(QPDFObjectHandle& qpdf_content)
  {
    LOG_S(INFO) << __FUNCTION__;

    qpdf_stream_decoder decoder(stream);
    decoder.decode(qpdf_content);
  }

  void pdf_decoder<STREAM>::interprete(std::vector<qpdf_instruction>& parameters)
  {
    LOG_S(INFO) << __FUNCTION__;

    // initialise the stack
    if(stack.size()==0)
      {
        //stack.clear();
        
        pdf_state<GLOBAL> state(page_cells, page_lines, page_images, 
				page_fonts, page_grphs);
        stack.push_back(state);
      }

    interprete_stream(parameters);    
  }

  bool pdf_decoder<STREAM>::update_stack(std::vector<pdf_state<GLOBAL> >& stack_,
                                         int                              stack_count_)
  {
    stack       = stack_;
    stack_count = stack_count_;

    if(stack.size()>0 and page_fonts.keys()!=cgs().page_fonts.keys())
      {
        pdf_state<GLOBAL> state(page_cells, page_lines, page_images, 
				page_fonts, page_grphs);
        state = stack.back();

        stack.push_back(state);              

        return true;
      }

    return false;
  }

  void pdf_decoder<STREAM>::interprete(std::vector<qpdf_instruction>& stream_,
                                       std::vector<qpdf_instruction>& parameters_)
  {
    LOG_S(INFO) << __FUNCTION__;

    stream = stream_;

    interprete_stream(parameters_);

    if(parameters_.size()!=0)
      {
        LOG_S(ERROR) << "Finishing a `Do` with nonzero number of parameters!";
      }
  }

  void pdf_decoder<STREAM>::interprete_stream(std::vector<qpdf_instruction>& parameters)
  {
    LOG_S(INFO) << __FUNCTION__;
    bool const profile_enabled = td_stream_profile_enabled();

    //assert(page_fonts.keys()==cgs().page_fonts.keys());

    for(int l=0; l<stream.size(); l++)
      {
        qpdf_instruction& inst = stream[l];
        
        if(inst.key=="operator")
          {
            pdf_operator::operator_name  name = pdf_operator::to_name(inst.val);
            pdf_operator::operator_class clss = pdf_operator::to_class(name);
            if(profile_enabled)
              {
                td_stream_profile().operator_count += 1;
              }

            // Skip path drawing operators — they produce lines/rectangles,
            // not text cells. Safe to skip for text extraction.
            if(clss==pdf_operator::PATH_CONSTRUCTION or
               clss==pdf_operator::PATH_PAINTING)
              {
                if(profile_enabled)
                  {
                    td_stream_profile().path_skip_count += 1;
                  }
                parameters.clear();
                continue;
              }

            execute_operator(inst, parameters);

            parameters.clear();
          }
        else
          {
            parameters.push_back(inst);
          }
      }
  }

  // get current global state
  pdf_state<GLOBAL>& pdf_decoder<STREAM>::cgs()
  {
    if(stack.size()==0)
      {
	std::stringstream message;
	message << "stack-size is zero in " << __FILE__ << ":" << __LINE__;

	LOG_S(ERROR) << message.str();
	throw std::logic_error(message.str());
      }

    pdf_state<GLOBAL>& state = stack.back();
    return state;
  }
 
  // get current text state
  pdf_state<TEXT>& pdf_decoder<STREAM>::cts()
  {
    return cgs().text_state;
  } 

  // get current text state
  pdf_state<LINE>& pdf_decoder<STREAM>::cls()
  {
    return cgs().line_state;
  } 

  // get current graphics state
  pdf_state<GRPH>& pdf_decoder<STREAM>::cgrs()
  {
    return cgs().grph_state;
  } 

  /*  
  void pdf_decoder<STREAM>::q()
  {
    pdf_state<GLOBAL> state(page_cells, page_lines, page_images, page_fonts);
    
    if(stack.size()>0)
      {
        state = stack.back();
      }
    
    stack.push_back(state);    

    stack_count += 1;
  } 
  */

  void pdf_decoder<STREAM>::q()
  {
    if(stack.size()==0)
      {
        pdf_state<GLOBAL> state(page_cells, page_lines, page_images, 
				page_fonts, page_grphs);
        stack.push_back(state);      
      }
    else
      {
        pdf_state<GLOBAL> state(stack.back());
        stack.push_back(state);      
      }

    stack_count += 1;
  } 

  void pdf_decoder<STREAM>::Q()
  {
    if(stack.size()>0)
      {
        stack.pop_back();
      }
    else
      {
        LOG_S(ERROR) << "invoking 'Q' on empty stack!";
        //throw std::logic_error(__FILE__);
      }    
  } 

  void pdf_decoder<STREAM>::execute_operator(const qpdf_instruction&              op,
                                             std::vector<qpdf_instruction>& parameters)
  {
    pdf_operator::operator_name name = pdf_operator::to_name(op.val); 

    switch(name)
      {

        /**************************************************
         ***  General graphics state
         **************************************************/

      case pdf_operator::w:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cgrs().w(parameters);
        }
        break;

      case pdf_operator::J:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cgrs().J(parameters);
        }
        break;

      case pdf_operator::j:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cgrs().j(parameters);
        }
        break;

      case pdf_operator::M:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cgrs().M(parameters);
        }
        break;

      case pdf_operator::d:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cgrs().d(parameters);
        }
        break;

      case pdf_operator::ri:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cgrs().ri(parameters);
        }
        break;

      case pdf_operator::i:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cgrs().i(parameters);
        }
        break;

      case pdf_operator::gs:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cgrs().gs(parameters);
        }
        break;

        /**************************************************
         ***  Special graphics state
         **************************************************/

      case pdf_operator::q:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          this->q();
        }
        break;

      case pdf_operator::Q:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          this->Q();
        }
        break;        

      case pdf_operator::cm: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cgs().cm(parameters);
        }
        break;

        /**************************************************
         ***  XObjects
         **************************************************/

      case pdf_operator::Do:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          bool const profile_enabled = td_stream_profile_enabled();
          auto const do_start = profile_enabled ? td_stream_profile_clock::now() : td_stream_profile_clock::time_point{};
          if(profile_enabled)
            {
              td_stream_profile().do_count += 1;
            }

          std::string xobj_name = parameters[0].to_utf8_string();

          if(not page_xobjects.has(xobj_name))
            {
              LOG_S(ERROR) << "unknown xobject with name `" << xobj_name << "`";
              return;
            }

          pdf_resource<PAGE_XOBJECT>& xobj = page_xobjects.get(xobj_name);

          switch(xobj.get_subtype())
            {
            case XOBJECT_IMAGE:
              {
                LOG_S(INFO) << "Do_Image: image with `" << xobj_name << "`";
                if(profile_enabled)
                  {
                    td_stream_profile().do_image_count += 1;
                  }
                cgs().Do_image(xobj);
              }
              break;

            case XOBJECT_FORM:
              {
                LOG_S(INFO) << "Do_Form: XObject with name `" << xobj_name << "`";
                if(profile_enabled)
                  {
                    td_stream_profile().do_form_count += 1;
                  }

                auto const copy_start =
                  profile_enabled ? td_stream_profile_clock::now() : td_stream_profile_clock::time_point{};
                bool const force_copy_resources = td_stream_force_copy_resources();
                std::pair<nlohmann::json, QPDFObjectHandle> xobj_fonts = xobj.get_fonts();
                std::pair<nlohmann::json, QPDFObjectHandle> xobj_grphs = xobj.get_grphs();
                std::pair<nlohmann::json, QPDFObjectHandle> xobj_xobjects = xobj.get_xobjects();

                pdf_resource<PAGE_FONTS>    page_fonts_;
                pdf_resource<PAGE_GRPHS>    page_grphs_;
                pdf_resource<PAGE_XOBJECTS> page_xobjects_;

                pdf_resource<PAGE_FONTS>*    nested_page_fonts = &page_fonts;
                pdf_resource<PAGE_GRPHS>*    nested_page_grphs = &page_grphs;
                pdf_resource<PAGE_XOBJECTS>* nested_page_xobjects = &page_xobjects;

                // parse the resources of the xobject
                {
                  if(force_copy_resources or not xobj_fonts.first.empty())
                    {
                      page_fonts_ = page_fonts;
                      if(not xobj_fonts.first.empty())
                        {
                          page_fonts_.set(xobj_fonts.first, xobj_fonts.second);
                        }
                      nested_page_fonts = &page_fonts_;
                    }

                  if(force_copy_resources or not xobj_grphs.first.empty())
                    {
                      page_grphs_ = page_grphs;
                      if(not xobj_grphs.first.empty())
                        {
                          page_grphs_.set(xobj_grphs.first, xobj_grphs.second);
                        }
                      nested_page_grphs = &page_grphs_;
                    }

                  if(force_copy_resources or not xobj_xobjects.first.empty())
                    {
                      page_xobjects_ = page_xobjects;
                      if(not xobj_xobjects.first.empty())
                        {
                          page_xobjects_.set(xobj_xobjects.first, xobj_xobjects.second);
                        }
                      nested_page_xobjects = &page_xobjects_;
                    }
                }
                if(profile_enabled)
                  {
                    td_stream_profile().do_copy_resources_ns +=
                      td_stream_profile_ns(td_stream_profile_clock::now() - copy_start);
                  }
                
                {
                  // push-back the stack 
                  this->q();

                  // transform coordinate system
                  cgs().cm(xobj.get_matrix());

                  {
                    auto const parse_stream_start =
                      profile_enabled ? td_stream_profile_clock::now() : td_stream_profile_clock::time_point{};
                    std::vector<qpdf_instruction> insts = xobj.parse_stream();
                    if(profile_enabled)
                      {
                        td_stream_profile().do_parse_stream_ns +=
                          td_stream_profile_ns(td_stream_profile_clock::now() - parse_stream_start);
                      }

                    pdf_decoder<STREAM> new_stream(page_dimension, page_cells, 
                                                   page_lines, page_images, 
                                                   *nested_page_fonts, *nested_page_grphs, 
						   *nested_page_xobjects);

                    auto const update_stack_start =
                      profile_enabled ? td_stream_profile_clock::now() : td_stream_profile_clock::time_point{};
                    bool updated_stack = new_stream.update_stack(stack, stack_count);
                    if(profile_enabled)
                      {
                        td_stream_profile().do_update_stack_ns +=
                          td_stream_profile_ns(td_stream_profile_clock::now() - update_stack_start);
                      }

                    // copy the stack
                    std::vector<qpdf_instruction> parameters;
                    auto const nested_interpret_start =
                      profile_enabled ? td_stream_profile_clock::now() : td_stream_profile_clock::time_point{};
                    new_stream.interprete(insts, parameters);
                    if(profile_enabled)
                      {
                        td_stream_profile().do_nested_interpret_ns +=
                          td_stream_profile_ns(td_stream_profile_clock::now() - nested_interpret_start);
                      }

                    if(updated_stack)
                      {
                        new_stream.Q();
                      }

                    auto unkown_ops = new_stream.get_unknown_operators();
                    for(auto item:unkown_ops)
                      {
                        unknown_operators.insert(item);
                      }
                  }

                  // pop-back the stack 
                  this->Q();                
                }

                LOG_S(INFO) << "ending the execution of FORM XObject with name" << xobj_name;
              }
              break;

            default:
              {
                LOG_S(ERROR) << " unknown subtype of xobject with name " << xobj_name;
              }
            }

          if(profile_enabled)
            {
              td_stream_profile().do_total_ns +=
                td_stream_profile_ns(td_stream_profile_clock::now() - do_start);
            }
        }
        break;

        /**************************************************
         ***  color-schemes
         **************************************************/

      case pdf_operator::CS: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().CS(parameters);
        }
        break;

      case pdf_operator::cs: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().cs(parameters);
        }
        break;

      case pdf_operator::SC: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().SC(parameters);
        }
        break;

      case pdf_operator::SCN: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().SCN(parameters);
        }
        break;

      case pdf_operator::sc: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().sc(parameters);
        }
        break;

      case pdf_operator::scn: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().scn(parameters);
        }
        break;

      case pdf_operator::G: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().G(parameters);
        }
        break;

      case pdf_operator::g:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().g(parameters);
        }
        break;

      case pdf_operator::RG: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().RG(parameters);
        }
        break;

      case pdf_operator::rg:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().rg(parameters);
        }
        break;
	
      case pdf_operator::K: 
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().K(parameters);
        }
        break;

      case pdf_operator::k:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
	  cgrs().k(parameters);
        }
        break;
	
        /**************************************************
         ***  text-objects
         **************************************************/

      case pdf_operator::BT:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          assert(page_fonts.keys()==cgs().page_fonts.keys());

          cts().BT();
        }
        break;

      case pdf_operator::ET:
        {
          LOG_S(INFO) << "executing " << to_string(name);
          cts().ET();
        }
        break;

        /**************************************************
         ***  text-state
         **************************************************/

      case pdf_operator::Tc:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Tc(parameters);
        }
        break;

      case pdf_operator::Tw:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Tw(parameters);
        }
        break;

      case pdf_operator::Tz:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Tz(parameters);
        }
        break;

      case pdf_operator::TL:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().TL(parameters);
        }
        break;

      case pdf_operator::Tf:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Tf(parameters);
        }
        break;

      case pdf_operator::Tr:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Tr(parameters);
        }
        break;

      case pdf_operator::Ts:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Ts(parameters);
        }
        break;

        /**************************************************
         ***  text-positioning
         **************************************************/

      case pdf_operator::Td:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Td(parameters);
        }
        break;

      case pdf_operator::TD:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().TD(parameters);
        }
        break;

      case pdf_operator::Tm:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Tm(parameters);
        }
        break;

      case pdf_operator::TStar:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().TStar(parameters);
        }
        break;

        /**************************************************
         ***  text-showing
         **************************************************/

      case pdf_operator::Tj:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().Tj(parameters, stack_count);
        }
        break;

      case pdf_operator::TJ:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cts().TJ(parameters, stack_count);
        }
        break;

      case pdf_operator::accent:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          assert(parameters.size()==1);

          std::vector<qpdf_instruction> TStar_params = {};
          cts().TStar(TStar_params);
          
          std::vector<qpdf_instruction> Tj_params = {parameters[0]};          
          cts().Tj(Tj_params, stack_count);          
        }
        break;

      case pdf_operator::double_accent:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          assert(parameters.size()==3);

          std::vector<qpdf_instruction> Tw_params = {parameters[0]};
          cts().Tw(Tw_params);
          
          std::vector<qpdf_instruction> Tc_params = {parameters[1]};
          cts().Tc(Tc_params);

          std::vector<qpdf_instruction> TStar_params = {};
          cts().TStar(TStar_params);

          std::vector<qpdf_instruction> Tj_params = {parameters[2]};          
          cts().Tj(Tj_params, stack_count);          
        }
        break;

        /**************************************************
         ***  paths construction [page 132-133]
         **************************************************/

      case pdf_operator::m:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().m(parameters);
        }
        break;

      case pdf_operator::l:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().l(parameters);
        }
        break;

      case pdf_operator::c:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().c(parameters);
        }
        break;

      case pdf_operator::v:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().v(parameters);
        }
        break;

      case pdf_operator::y:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().y(parameters);
        }
        break;

      case pdf_operator::h:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().h(parameters);
        }
        break;

      case pdf_operator::re:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().re(parameters);          
        }
        break;

        /**************************************************
         ***  path painting [page 132-133]
         **************************************************/

      case pdf_operator::s:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().s(parameters);          
        }
        break;

      case pdf_operator::S:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().S(parameters);          
        }
        break;

      case pdf_operator::f:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().f(parameters);          
        }
        break;

      case pdf_operator::F:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().F(parameters);          
        }
        break;

      case pdf_operator::fStar:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().fStar(parameters);          
        }
        break;

      case pdf_operator::B:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().B(parameters);          
        }
        break;

      case pdf_operator::BStar:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().BStar(parameters);          
        }
        break;

      case pdf_operator::b:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().b(parameters);          
        }
        break;

      case pdf_operator::bStar:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().bStar(parameters);          
        }
        break;

      case pdf_operator::n:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().n(parameters);          
        }
        break;

        /**************************************************
         ***  path clipping [page ...]
         **************************************************/

      case pdf_operator::W:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().W(parameters);          
        }
        break;

      case pdf_operator::WStar:
        {
          LOG_S(INFO) << "executing " << to_string(name);          
          cls().WStar(parameters);          
        }
        break;

        /**************************************************
         ***  other
         **************************************************/

      case pdf_operator::null:
        {
          LOG_S(WARNING) << "unknown operator with name: " << op.val;
          unknown_operators.insert(op.val);
        }
        break;

      default:
        {
          LOG_S(WARNING) << "ignored operator with name: " << op.val;          
          unknown_operators.insert(op.val);          
        }        
      }
  }

}

#endif
