//-*-C++-*-

#ifndef QPDF_STREAM_DECODER_H
#define QPDF_STREAM_DECODER_H

#include <chrono>
#include <exception>
#include <cstdio>
#include <cstdlib>

namespace
{
  using qpdf_profile_clock = std::chrono::steady_clock;

  inline bool qpdf_profile_enabled()
  {
    static bool enabled = []() {
      if(auto const* env = std::getenv("TD_QPDF_PROFILE"))
        {
          return (env[0] != '\0') && !((env[0] == '0') && (env[1] == '\0'));
        }
      return false;
    }();

    return enabled;
  }

  inline long long qpdf_profile_ns(qpdf_profile_clock::duration duration)
  {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
  }

  inline double qpdf_profile_ms(long long ns)
  {
    return static_cast<double>(ns) / 1000000.0;
  }
}

namespace pdflib
{

  class qpdf_stream_decoder:
    public QPDFObjectHandle::ParserCallbacks
  {
  public:

    qpdf_stream_decoder(std::vector<qpdf_instruction>& stream_);
    ~qpdf_stream_decoder();

    void print();

    void decode(QPDFObjectHandle& content);

    // methods used by the QPDFObjectHandle::ParserCallbacks
    ///*virtual*/ void handleObject(QPDFObjectHandle obj) override;

    void handleObject(QPDFObjectHandle obj, size_t offset, size_t len) override;
    
    void contentSize(size_t len) override;

    void handleEOF() override;
    
  private:

    std::vector<qpdf_instruction>& stream;

    std::regex value_pattern_0;

    bool profile_enabled;
    size_t profile_object_count;
    size_t profile_operator_count;
    size_t profile_number_count;
    size_t profile_null_count;
    long long profile_callback_ns;
    long long profile_stringify_ns;
    long long profile_regex_ns;
    long long profile_push_ns;
  };

  qpdf_stream_decoder::qpdf_stream_decoder(std::vector<qpdf_instruction>& stream_):
    QPDFObjectHandle::ParserCallbacks(),
    stream(stream_),

    value_pattern_0(R"(^(\d\.\d+)(\-\d+)$)"),
    profile_enabled(false),
    profile_object_count(0),
    profile_operator_count(0),
    profile_number_count(0),
    profile_null_count(0),
    profile_callback_ns(0),
    profile_stringify_ns(0),
    profile_regex_ns(0),
    profile_push_ns(0)
  {}

  qpdf_stream_decoder::~qpdf_stream_decoder()
  {}

  void qpdf_stream_decoder::print()
  {
    for(auto row:stream)
      {
        LOG_S(INFO) << std::setw(12) << row.key << " | " << row.val;
      }
  }

  void qpdf_stream_decoder::decode(QPDFObjectHandle& content)
  {
    LOG_S(INFO) << "start decoding content-stream: " << content.getTypeName() << " -> " << content.unparse();

    stream.clear();
    profile_enabled = qpdf_profile_enabled();
    profile_object_count = 0;
    profile_operator_count = 0;
    profile_number_count = 0;
    profile_null_count = 0;
    profile_callback_ns = 0;
    profile_stringify_ns = 0;
    profile_regex_ns = 0;
    profile_push_ns = 0;
    auto const decode_start = profile_enabled ? qpdf_profile_clock::now() : qpdf_profile_clock::time_point{};

    try
      {
        QPDFObjectHandle::parseContentStream(content, this);
      }
    catch(std::exception& e)
      {
        LOG_S(ERROR) << "QPDF encountered error (" << e.what() << ") during decoding";
      }

    if(profile_enabled)
      {
        long long const total_ns = qpdf_profile_ns(qpdf_profile_clock::now() - decode_start);
        std::fprintf(
          stderr,
          "[td-stream-decoder] objects=%zu operators=%zu numbers=%zu nulls=%zu "
          "stringify=%.3fms regex=%.3fms push=%.3fms callback=%.3fms total=%.3fms\n",
          profile_object_count,
          profile_operator_count,
          profile_number_count,
          profile_null_count,
          qpdf_profile_ms(profile_stringify_ns),
          qpdf_profile_ms(profile_regex_ns),
          qpdf_profile_ms(profile_push_ns),
          qpdf_profile_ms(profile_callback_ns),
          qpdf_profile_ms(total_ns));
      }

    LOG_S(WARNING) << "finished decoding content-stream!";
  }

  /*
  void qpdf_stream_decoder::handleObject(QPDFObjectHandle obj)
  {
    LOG_S(INFO) << __FUNCTION__;

    qpdf_instruction row;
    {
      row.key = obj.getTypeName();
      row.val = obj.unparse();
      row.obj = obj;

      //LOG_S(INFO) << std::setw(12) << row.key << " | " << row.val;
    }

    stream.push_back(row);
  }
  */

  void qpdf_stream_decoder::handleObject(QPDFObjectHandle obj, size_t offset, size_t len)
  {
    auto const callback_start = profile_enabled ? qpdf_profile_clock::now() : qpdf_profile_clock::time_point{};
    qpdf_instruction row;
    row.obj = obj;

    if(obj.isNull())
      {
        if(profile_enabled)
          {
            ++profile_null_count;
          }
        // Reinterpret null as empty array (workaround for 'd' operator, Table 56)
        row.key = "parameter";
        row.val = "[]";
      }
    else if(obj.isOperator())
      {
        if(profile_enabled)
          {
            ++profile_operator_count;
          }
        // Operators: need key + val for dispatch, but skip regex
        auto const stringify_start = profile_enabled ? qpdf_profile_clock::now() : qpdf_profile_clock::time_point{};
        row.key = obj.getTypeName();
        row.val = obj.unparse();
        if(profile_enabled)
          {
            profile_stringify_ns += qpdf_profile_ns(qpdf_profile_clock::now() - stringify_start);
          }
      }
    else
      {
        bool const is_number = obj.isNumber();
        if(profile_enabled && is_number)
          {
            ++profile_number_count;
          }

        auto const stringify_start = profile_enabled ? qpdf_profile_clock::now() : qpdf_profile_clock::time_point{};
        row.key = obj.getTypeName();
        row.val = obj.unparse();
        if(profile_enabled)
          {
            profile_stringify_ns += qpdf_profile_ns(qpdf_profile_clock::now() - stringify_start);
          }

        // Only run the malformed-float regex on real/integer tokens that
        // contain a dash (extremely rare edge case: "1.234-5")
        if(is_number && row.val.find('-') != std::string::npos)
          {
            auto const regex_start = profile_enabled ? qpdf_profile_clock::now() : qpdf_profile_clock::time_point{};
            std::smatch match;
            if(std::regex_match(row.val, match, value_pattern_0))
              {
                LOG_S(WARNING) << std::setw(12) << row.key << " | " << row.val
                               << " => new matched value: " << match[1];

                double value = std::stod(match[1].str());
                QPDFObjectHandle new_obj = QPDFObjectHandle::newReal(value);

                row.key = new_obj.getTypeName();
                row.val = new_obj.unparse();
                row.obj = new_obj;
              }
            if(profile_enabled)
              {
                profile_regex_ns += qpdf_profile_ns(qpdf_profile_clock::now() - regex_start);
              }
          }
      }

    auto const push_start = profile_enabled ? qpdf_profile_clock::now() : qpdf_profile_clock::time_point{};
    stream.push_back(std::move(row));
    if(profile_enabled)
      {
        profile_push_ns += qpdf_profile_ns(qpdf_profile_clock::now() - push_start);
        ++profile_object_count;
        profile_callback_ns += qpdf_profile_ns(qpdf_profile_clock::now() - callback_start);
      }
  }

  void qpdf_stream_decoder::contentSize(size_t len)
  {
    //LOG_S(INFO) << __FUNCTION__ << ": " << len;
  }

  void qpdf_stream_decoder::handleEOF()
  {
    //LOG_S(INFO) << __FUNCTION__;
  }

}

#endif
